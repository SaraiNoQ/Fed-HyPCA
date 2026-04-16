#!/bin/bash
# Sanity check: 2-org toy experiment with minimal compute.
# Should complete in <1 hour on a single GPU.
# Validates: training loop runs, metrics compute, constraints work.

set -e

MODEL="/root/autodl-tmp/model"
RESULTS_DIR="results/sanity"
OUTPUT_DIR="outputs/sanity"

mkdir -p "$RESULTS_DIR" "$OUTPUT_DIR"

echo "=========================================="
echo "Sanity Check: 2-org, 5 rounds, small data"
echo "=========================================="

# Minimal FedAvg (should show high AggViol)
echo ">>> FedAvg (expect high violation)"
python train_federated.py \
    --model_name "$MODEL" \
    --aggregation fedavg \
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
    --experiment_name "sanity_fedavg" \
    --output_dir "$OUTPUT_DIR" \
    --results_dir "$RESULTS_DIR" \
    --eval_every 5

# Minimal Fed-HyPCA (should show lower AggViol)
echo ">>> Fed-HyPCA (expect lower violation)"
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
    --experiment_name "sanity_fedhypca" \
    --output_dir "$OUTPUT_DIR" \
    --results_dir "$RESULTS_DIR" \
    --eval_every 5

echo ""
echo "=========================================="
echo "Sanity check complete! Compare results:"
echo "  $RESULTS_DIR/sanity_fedavg_final.json"
echo "  $RESULTS_DIR/sanity_fedhypca_final.json"
echo "=========================================="
