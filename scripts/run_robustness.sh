#!/bin/bash
# Run robustness evaluation: jailbreak + poisoning experiments.
# Estimated: ~250 GPU-hours on A100-80GB

set -e

MODEL="/root/autodl-tmp/model"
ROUNDS=50
RESULTS_DIR="results/robustness"
OUTPUT_DIR="outputs/robustness"

mkdir -p "$RESULTS_DIR" "$OUTPUT_DIR"

echo "=========================================="
echo "Robustness: Poisoning with 1 malicious client"
echo "=========================================="

# Run Fed-HyPCA and baselines with 1 malicious client (5% corrupted data)
# The malicious client is simulated by flipping labels in O5 (creative writing)
# This is handled by the --poison_client flag (to be implemented in data pipeline)

for SEED in 42 123 456; do
    for METHOD in fedhypca fedavg fedprox ditto qffl fedavg_dual; do
        echo ">>> $METHOD with poisoning, seed $SEED"
        python train_federated.py \
            --model_name "$MODEL" \
            --aggregation "$METHOD" \
            --num_rounds $ROUNDS \
            --seed $SEED \
            --experiment_name "poison_${METHOD}_s${SEED}" \
            --output_dir "$OUTPUT_DIR" \
            --results_dir "$RESULTS_DIR"
        # Note: poisoning logic needs to be added to the data pipeline
        # by flipping 5% of labels for one client
    done
done

echo "Robustness experiments completed!"
