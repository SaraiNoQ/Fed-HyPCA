#!/bin/bash
# Validation experiment for Fed-HyPCA
# Model: Qwen3.5-4B-base
# Rounds: 30, eval every 5 rounds
# Reduced dataset (6 orgs, 3k samples/org) for faster training
# Memory-optimized for 32GB GPU
#
# Key fixes applied:
# - rho=0.01 (was 0.1): allow client personalization
# - eta_lambda=0.05 (was 0.01): faster dual response
# - scalar_reweight_beta=5.0: exponential violation reweighting
# - dual_update_interval=10: stable dual updates

set -e

MODEL="/home/qiqi/models/qwen3.5-4b"
ROUNDS=30
RESULTS_DIR="results/validation_v2"
OUTPUT_DIR="outputs/validation_v2"

mkdir -p "$RESULTS_DIR" "$OUTPUT_DIR"

echo "=========================================="
echo "Fed-HyPCA Validation Experiment (v2)"
echo "Model: Qwen3.5-4B-base"
echo "Rounds: $ROUNDS"
echo "Data: 3000 samples/org (reduced for speed)"
echo "Eval every: 5 rounds"
echo "Key: rho=0.01, eta_lambda=0.05, scalar_beta=5.0"
echo "=========================================="

# Run Fed-HyPCA first
echo ""
echo ">>> Running Fed-HyPCA (main method)"
python -u train_federated.py \
    --model_name "$MODEL" \
    --aggregation fedhypca \
    --num_rounds $ROUNDS \
    --local_epochs 1 \
    --local_steps_per_epoch 500 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --rho 0.01 \
    --eta_lambda 0.05 \
    --eta_nu 0.1 \
    --beta_b 25 \
    --scalar_reweight_beta 5.0 \
    --dual_update_interval 10 \
    --train_size_per_org 3000 \
    --val_size_per_org 500 \
    --test_size_per_org 500 \
    --seed 42 \
    --experiment_name "fedhypca_validation_v2_s42" \
    --output_dir "$OUTPUT_DIR" \
    --results_dir "$RESULTS_DIR" \
    --eval_every 5 \
    --save_every 10

echo ""
echo "=========================================="
echo "Fed-HyPCA completed! Now running FedAvg baseline..."
echo "=========================================="

# Run FedAvg baseline with same training config (but no constraints/proximal)
echo ""
echo ">>> Running FedAvg (baseline)"
python -u train_federated.py \
    --model_name "$MODEL" \
    --aggregation fedavg \
    --num_rounds $ROUNDS \
    --local_epochs 1 \
    --local_steps_per_epoch 500 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --train_size_per_org 3000 \
    --val_size_per_org 500 \
    --test_size_per_org 500 \
    --seed 42 \
    --experiment_name "fedavg_baseline_v2_s42" \
    --output_dir "$OUTPUT_DIR" \
    --results_dir "$RESULTS_DIR" \
    --eval_every 5 \
    --save_every 10

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
