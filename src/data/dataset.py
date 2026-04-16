"""Dataset loading and per-org splitting for Fed-HyPCA benchmark."""

import json
import os
import pickle
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from src.data.taxonomy import (
    BEAVERTAILS_MAPPING,
    SAFETY_CATEGORIES,
    get_org_must_refuse_mask,
    get_all_org_ids,
    should_refuse,
)
from configs.default import ORG_POLICIES


class OrgAlignmentDataset(Dataset):
    """Dataset for a single organization's alignment training.

    Each sample has:
    - prompt: the input text
    - response: the target response (for allowed prompts) or refusal template
    - category_idx: which safety category this belongs to (-1 for benign)
    - is_unsafe: whether this prompt is unsafe according to THIS org's policy
    - label: 1 if should refuse, 0 if should respond
    """

    def __init__(self, samples: list[dict], org_id: str, tokenizer=None, max_length: int = 1024):
        self.samples = samples
        self.org_id = org_id
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.must_refuse_mask = get_org_must_refuse_mask(org_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        category_idx = sample.get("category_idx", -1)

        # Determine if this org should refuse this prompt
        if category_idx >= 0 and self.must_refuse_mask[category_idx]:
            label = 1  # refuse
        else:
            label = 0  # respond

        item = {
            "prompt": sample["prompt"],
            "response": sample.get("response", ""),
            "category_idx": category_idx,
            "label": label,
            "org_id": self.org_id,
        }

        if self.tokenizer is not None:
            encoded = self.tokenizer(
                sample["prompt"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            item["input_ids"] = encoded["input_ids"].squeeze(0)
            item["attention_mask"] = encoded["attention_mask"].squeeze(0)

        return item


def load_beavertails(cache_dir: str = "data/cache", split: str = "330k_train", use_cache: bool = True) -> list[dict]:
    """Load BeaverTails dataset and map to our taxonomy.

    Uses pickle cache to speed up subsequent loads.
    """
    cache_file = os.path.join(cache_dir, "beavertails_processed.pkl")

    # Try to load from cache
    if use_cache and os.path.exists(cache_file):
        print(f"  Loading from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # Use HF mirror for China if needed
    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    ds = load_dataset("PKU-Alignment/BeaverTails", split=split, cache_dir=cache_dir)
    samples = []
    for item in ds:
        if not item.get("is_safe", True):
            # Map BeaverTails categories to our taxonomy
            bt_category = item.get("category", {})
            mapped_cat = None
            if isinstance(bt_category, dict):
                for cat_name, is_present in bt_category.items():
                    if is_present and cat_name in BEAVERTAILS_MAPPING:
                        mapped_cat = BEAVERTAILS_MAPPING[cat_name]
                        break
            elif isinstance(bt_category, str) and bt_category in BEAVERTAILS_MAPPING:
                mapped_cat = BEAVERTAILS_MAPPING[bt_category]

            if mapped_cat is not None:
                cat_idx = SAFETY_CATEGORIES.index(mapped_cat)
                samples.append({
                    "prompt": item["prompt"],
                    "response": item.get("response", ""),
                    "category_idx": cat_idx,
                    "source": "beavertails",
                    "is_unsafe_global": True,
                })

    # Save to cache
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(samples, f)
        print(f"  Saved to cache: {cache_file}")

    return samples


def load_hh_rlhf(cache_dir: str = "data/cache", split: str = "train", use_cache: bool = True) -> list[dict]:
    """Load HH-RLHF helpful split as benign/allowed prompts.

    Uses pickle cache to speed up subsequent loads.
    """
    cache_file = os.path.join(cache_dir, "hh_rlhf_processed.pkl")

    # Try to load from cache
    if use_cache and os.path.exists(cache_file):
        print(f"  Loading from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    ds = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)
    samples = []
    for item in ds:
        # Extract prompt from the chosen response
        chosen = item.get("chosen", "")
        if "\n\nHuman:" in chosen and "\n\nAssistant:" in chosen:
            parts = chosen.split("\n\nHuman:")
            if len(parts) >= 2:
                prompt_part = parts[1].split("\n\nAssistant:")[0].strip()
                response_part = parts[1].split("\n\nAssistant:")[-1].strip() if "\n\nAssistant:" in parts[1] else ""
                samples.append({
                    "prompt": prompt_part,
                    "response": response_part,
                    "category_idx": -1,  # benign
                    "source": "hh_rlhf",
                    "is_unsafe_global": False,
                })

    # Save to cache
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(samples, f)
        print(f"  Saved to cache: {cache_file}")

    return samples


def create_org_datasets(
    unsafe_samples: list[dict],
    benign_samples: list[dict],
    train_size: int = 12000,
    val_size: int = 1500,
    test_size: int = 2000,
    seed: int = 42,
) -> dict[str, dict[str, list[dict]]]:
    """Create per-org train/val/test splits.

    Each org gets:
    - A mix of benign prompts (should respond) and unsafe prompts
    - Unsafe prompts are labeled per-org based on policy vectors
    - Same prompt can be "refuse" for O1 but "respond" for O5
    """
    rng = random.Random(seed)

    # Shuffle
    unsafe_pool = list(unsafe_samples)
    benign_pool = list(benign_samples)
    rng.shuffle(unsafe_pool)
    rng.shuffle(benign_pool)

    org_ids = get_all_org_ids()
    org_datasets = {}

    for org_id in org_ids:
        total_needed = train_size + val_size + test_size
        # ~40% unsafe, ~60% benign per org
        n_unsafe = int(total_needed * 0.4)
        n_benign = total_needed - n_unsafe

        # Sample with replacement if pool is smaller than needed
        if len(unsafe_pool) < n_unsafe:
            org_unsafe = rng.choices(unsafe_pool, k=n_unsafe)
        else:
            org_unsafe = rng.sample(unsafe_pool, n_unsafe)

        if len(benign_pool) < n_benign:
            org_benign = rng.choices(benign_pool, k=n_benign)
        else:
            org_benign = rng.sample(benign_pool, n_benign)

        all_samples = org_unsafe + org_benign
        rng.shuffle(all_samples)

        org_datasets[org_id] = {
            "train": all_samples[:train_size],
            "val": all_samples[train_size:train_size + val_size],
            "test": all_samples[train_size + val_size:train_size + val_size + test_size],
        }

    return org_datasets


def build_benchmark(
    data_config,
    tokenizer=None,
    seed: int = 42,
) -> dict[str, dict[str, OrgAlignmentDataset]]:
    """Build the full 6-org benchmark.

    Returns: {org_id: {"train": Dataset, "val": Dataset, "test": Dataset}}
    """
    print("Loading BeaverTails (unsafe prompts)...")
    unsafe_samples = load_beavertails(cache_dir=data_config.cache_dir)
    print(f"  Loaded {len(unsafe_samples)} unsafe samples")

    print("Loading HH-RLHF (benign prompts)...")
    benign_samples = load_hh_rlhf(cache_dir=data_config.cache_dir)
    print(f"  Loaded {len(benign_samples)} benign samples")

    print("Creating per-org splits...")
    org_raw = create_org_datasets(
        unsafe_samples=unsafe_samples,
        benign_samples=benign_samples,
        train_size=data_config.train_size_per_org,
        val_size=data_config.val_size_per_org,
        test_size=data_config.test_size_per_org,
        seed=seed,
    )

    org_datasets = {}
    for org_id, splits in org_raw.items():
        org_datasets[org_id] = {}
        for split_name, samples in splits.items():
            org_datasets[org_id][split_name] = OrgAlignmentDataset(
                samples=samples,
                org_id=org_id,
                tokenizer=tokenizer,
                max_length=getattr(data_config, "max_seq_length", 1024),
            )
            n_refuse = sum(
                1 for s in samples
                if s.get("category_idx", -1) >= 0
                and should_refuse(org_id, s["category_idx"])
            )
            print(f"  {org_id}/{split_name}: {len(samples)} samples, {n_refuse} should-refuse")

    return org_datasets
