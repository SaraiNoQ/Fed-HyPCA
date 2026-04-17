#!/bin/bash
# Run full Fed-HyPCA method with 3 seeds.
# Estimated: ~100 GPU-hours on A100-80GB

set -e

MODEL="/home/qiqi/models/qwen3.5-4b"
ROUNDS=50
RESULTS_DIR="results/main"
OUTPUT_DIR="outputs/main"

mkdir -p "$RESULTS_DIR" "$OUTPUT_DIR"

for SEED in 42 123 456; do
    echo "=========================================="
    echo "Fed-HyPCA: Seed $SEED"
    echo "=========================================="

    python train_federated.py \
        --model_name "$MODEL" \
        --aggregation fedhypca \
        --num_rounds $ROUNDS \
        --local_epochs 3 \
        --batch_size 4 \
        --learning_rate 2e-4 \
        --rho 0.01 \
        --gamma 0.1 \
        --tau 0.1 \
        --eta_lambda 0.05 \
        --eta_nu 0.1 \
        --beta 1.0 \
        --beta_b 50.0 \
        --seed $SEED \
        --experiment_name "fedhypca_s${SEED}" \
        --output_dir "$OUTPUT_DIR" \
        --results_dir "$RESULTS_DIR" \
        --eval_every 5 \
        --save_every 25

done

echo "Fed-HyPCA runs completed!"
