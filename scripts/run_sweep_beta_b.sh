#!/bin/bash
# Hyperparameter sweep for beta_b to generate Pareto curves
# Run on remote server

set -e

MODEL="/home/qiqi/models/qwen3.5-4b"
RESULTS_DIR="results/sweep_beta_b"
OUTPUT_DIR="outputs/sweep_beta_b"

mkdir -p "$RESULTS_DIR" "$OUTPUT_DIR"

echo "=========================================="
echo "Beta_b sweep for Pareto curves"
echo "=========================================="

# Sweep beta_b values: 10, 25, 50, 100, 200
for BETA_B in 10 25 50 100 200; do
    echo ""
    echo ">>> Fed-HyPCA with beta_b=$BETA_B"
    python train_federated.py \
        --model_name "$MODEL" \
        --aggregation fedhypca \
        --num_rounds 5 \
        --local_epochs 1 \
        --local_steps_per_epoch 5 \
        --num_orgs 2 \
        --train_size_per_org 100 \
        --val_size_per_org 25 \
        --test_size_per_org 50 \
        --batch_size 1 \
        --gradient_accumulation_steps 1 \
        --seed 42 \
        --beta_b $BETA_B \
        --experiment_name "sweep_beta_b_${BETA_B}" \
        --output_dir "$OUTPUT_DIR" \
        --results_dir "$RESULTS_DIR" \
        --eval_every 5
done

echo ""
echo "=========================================="
echo "Sweep complete! Results:"
for BETA_B in 10 25 50 100 200; do
    echo "  $RESULTS_DIR/sweep_beta_b_${BETA_B}_final.json"
done
echo "=========================================="
