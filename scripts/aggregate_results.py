"""Aggregate results across seeds and methods into comparison tables.

Usage:
    python scripts/aggregate_results.py --results_dir results/baselines results/main \
                                        --output results/comparison_table.csv
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def load_results(results_dir: str) -> list[dict]:
    """Load all JSON result files from a directory."""
    results = []
    for f in Path(results_dir).glob("*_final.json"):
        with open(f) as fp:
            data = json.load(fp)
            data["_file"] = str(f)
            # Extract method and seed from filename
            name = f.stem.replace("_final", "")
            parts = name.rsplit("_s", 1)
            if len(parts) == 2:
                data["_method"] = parts[0]
                data["_seed"] = int(parts[1])
            else:
                data["_method"] = name
                data["_seed"] = 0
            results.append(data)
    return results


def aggregate_across_seeds(results: list[dict]) -> pd.DataFrame:
    """Aggregate results across seeds, computing mean ± std."""
    method_results = defaultdict(list)

    for r in results:
        method = r["_method"]
        g = r.get("global", {})
        method_results[method].append({
            "agg_viol": g.get("agg_viol", None),
            "pers_viol": g.get("pers_viol", 0),
            "worst_viol": g.get("worst_viol", 0),
            "avg_ref_recall": g.get("avg_ref_recall", 0),
            "min_ref_recall": g.get("min_ref_recall", 0),
            "avg_over_refusal": g.get("avg_over_refusal", 0),
            "max_over_refusal": g.get("max_over_refusal", 0),
            "avg_utility_nll": g.get("avg_utility_nll", 0),
            "consensus_dist": g.get("consensus_dist", None),
        })

    rows = []
    for method, seeds in sorted(method_results.items()):
        row = {"Method": method, "Seeds": len(seeds)}
        for metric in seeds[0]:
            values = [s[metric] for s in seeds if s[metric] is not None]
            if values:
                mean = np.mean(values)
                std = np.std(values)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
                row[metric] = f"{mean:.4f}±{std:.4f}"
            else:
                row[metric] = "N/A"
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", nargs="+", required=True)
    parser.add_argument("--output", type=str, default="results/comparison_table.csv")
    args = parser.parse_args()

    all_results = []
    for d in args.results_dir:
        all_results.extend(load_results(d))

    if not all_results:
        print("No results found!")
        return

    print(f"Loaded {len(all_results)} result files")

    df = aggregate_across_seeds(all_results)

    # Print table
    display_cols = [
        "Method", "Seeds",
        "agg_viol", "pers_viol", "worst_viol",
        "avg_ref_recall", "min_ref_recall",
        "avg_over_refusal", "avg_utility_nll",
    ]
    existing_cols = [c for c in display_cols if c in df.columns]
    print("\n" + df[existing_cols].to_string(index=False))

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
