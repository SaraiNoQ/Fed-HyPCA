"""Standalone evaluation script.

Usage:
    python evaluate.py --checkpoint outputs/fedhypca_r50_s42/checkpoint_final.pt \
                       --split test --output results/fedhypca_eval.json
"""

# GPU must be configured before any torch import
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from src.utils.gpu import configure_gpu_from_args, get_device
configure_gpu_from_args(default=0)  # GPU 0 = RTX 5880 Ada

import argparse
import json
import os

import torch

from configs.default import ExperimentConfig, ModelConfig, DataConfig, EvalConfig
from src.utils.seed import set_seed
from src.models.lora_model import FedHyPCAModel, load_tokenizer
from src.data.dataset import build_benchmark
from src.evaluation.metrics import evaluate_all_orgs, save_results, print_results_table


def parse_args():
    parser = argparse.ArgumentParser(description="Fed-HyPCA Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="/home/qiqi/models/qwen3.5-4b")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    # Build model
    model_config = ModelConfig(model_name=args.model_name)
    model = FedHyPCAModel(model_config)

    # Load tokenizer and datasets
    tokenizer = load_tokenizer(args.model_name)
    data_config = DataConfig(cache_dir=args.cache_dir)
    org_datasets = build_benchmark(data_config=data_config, tokenizer=tokenizer, seed=args.seed)

    # Build eval config
    eval_config = EvalConfig(eval_batch_size=args.eval_batch_size)
    config = ExperimentConfig(model=model_config, data=data_config, eval=eval_config)

    # Evaluate
    global_state = checkpoint.get("global_state")
    client_states = checkpoint.get("client_states")

    results = evaluate_all_orgs(
        model=model,
        org_datasets=org_datasets,
        config=config,
        device=device,
        global_state=global_state,
        client_states=client_states,
        split_name=args.split,
    )

    print_results_table(results)

    # Save
    if args.output:
        save_results(results, args.output)
    else:
        output_path = args.checkpoint.replace(".pt", f"_eval_{args.split}.json")
        save_results(results, output_path)


if __name__ == "__main__":
    main()
