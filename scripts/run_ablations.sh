#!/bin/bash
# Run all 12 ablations (A1-A12) with 3 seeds each.
# Estimated: ~800 GPU-hours on A100-80GB

set -e

MODEL="/root/autodl-tmp/model"
ROUNDS=50
RESULTS_DIR="results/ablations"
OUTPUT_DIR="outputs/ablations"

mkdir -p "$RESULTS_DIR" "$OUTPUT_DIR"

# Common args for all ablations
COMMON="--model_name $MODEL --aggregation fedhypca --num_rounds $ROUNDS \
        --output_dir $OUTPUT_DIR --results_dir $RESULTS_DIR --eval_every 10"

for SEED in 42 123 456; do
    echo "=========================================="
    echo "Ablations: Seed $SEED"
    echo "=========================================="

    # A1: FedAvg server (no constrained aggregation QP)
    echo ">>> A1: No constrained aggregation"
    python train_federated.py $COMMON --seed $SEED \
        --no_constrained_aggregation \
        --experiment_name "A1_no_constrained_agg_s${SEED}"

    # A2: Scalar violation weights only (no Jacobian correction)
    echo ">>> A2: No Jacobian correction"
    python train_federated.py $COMMON --seed $SEED \
        --no_jacobian_correction \
        --experiment_name "A2_no_jacobian_s${SEED}"

    # A3: No slack variables
    echo ">>> A3: No slack variables"
    python train_federated.py $COMMON --seed $SEED \
        --no_slack_variables \
        --experiment_name "A3_no_slack_s${SEED}"

    # A4: Fixed penalty coefficients (no adaptive duals)
    echo ">>> A4: Fixed penalty"
    python train_federated.py $COMMON --seed $SEED \
        --no_adaptive_duals \
        --experiment_name "A4_fixed_penalty_s${SEED}"

    # A5: No proximal term
    echo ">>> A5: No proximal term"
    python train_federated.py $COMMON --seed $SEED \
        --no_proximal_term \
        --experiment_name "A5_no_proximal_s${SEED}"

    # A6: Shared-only (u_i = θ, no personalization)
    echo ">>> A6: Shared-only"
    python train_federated.py $COMMON --seed $SEED \
        --no_personalization \
        --experiment_name "A6_shared_only_s${SEED}"

    # A7: Local-only (no federation)
    echo ">>> A7: Local-only"
    python train_federated.py $COMMON --seed $SEED \
        --no_federation \
        --num_rounds 1 --local_epochs 9 \
        --experiment_name "A7_local_only_s${SEED}"

    # A8: No over-refusal constraint
    echo ">>> A8: No over-refusal constraint"
    python train_federated.py $COMMON --seed $SEED \
        --no_overrefusal_constraint \
        --experiment_name "A8_no_overrefusal_s${SEED}"

    # A9: No refusal head auxiliary loss
    echo ">>> A9: No refusal head aux"
    python train_federated.py $COMMON --seed $SEED \
        --no_refusal_head_aux \
        --experiment_name "A9_no_aux_s${SEED}"

    # A10: Single "unsafe" bit (no structured policy)
    echo ">>> A10: Single unsafe bit"
    python train_federated.py $COMMON --seed $SEED \
        --no_structured_policy \
        --experiment_name "A10_single_bit_s${SEED}"

    # A11: Text-only policy grounding (placeholder — needs separate implementation)
    echo ">>> A11: Text-only policy (skipped — needs text grounding module)"

    # A12: Frozen λ after warm-start (freeze after round 10)
    echo ">>> A12: Frozen duals after round 10"
    python train_federated.py $COMMON --seed $SEED \
        --freeze_duals_after 10 \
        --experiment_name "A12_frozen_duals_s${SEED}"

done

echo "All ablations completed!"
