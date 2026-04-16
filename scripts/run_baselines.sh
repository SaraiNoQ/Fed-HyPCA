#!/bin/bash
# Run all baseline methods (B1-B10) with 3 seeds each.
# Estimated: ~700 GPU-hours on A100-80GB

set -e

MODEL="/root/autodl-tmp/model"
ROUNDS=50
RESULTS_DIR="results/baselines"
OUTPUT_DIR="outputs/baselines"

mkdir -p "$RESULTS_DIR" "$OUTPUT_DIR"

for SEED in 42 123 456; do
    echo "=========================================="
    echo "Seed: $SEED"
    echo "=========================================="

    # B1: FedAvg + LoRA
    echo ">>> B1: FedAvg"
    python train_federated.py \
        --model_name "$MODEL" \
        --aggregation fedavg \
        --num_rounds $ROUNDS \
        --seed $SEED \
        --experiment_name "fedavg_s${SEED}" \
        --output_dir "$OUTPUT_DIR" \
        --results_dir "$RESULTS_DIR"

    # B2: FedProx + LoRA
    echo ">>> B2: FedProx"
    python train_federated.py \
        --model_name "$MODEL" \
        --aggregation fedprox \
        --fedprox_mu 0.01 \
        --num_rounds $ROUNDS \
        --seed $SEED \
        --experiment_name "fedprox_s${SEED}" \
        --output_dir "$OUTPUT_DIR" \
        --results_dir "$RESULTS_DIR"

    # B3: SCAFFOLD + LoRA
    echo ">>> B3: SCAFFOLD"
    python train_federated.py \
        --model_name "$MODEL" \
        --aggregation scaffold \
        --num_rounds $ROUNDS \
        --seed $SEED \
        --experiment_name "scaffold_s${SEED}" \
        --output_dir "$OUTPUT_DIR" \
        --results_dir "$RESULTS_DIR"

    # B4: Ditto + LoRA
    echo ">>> B4: Ditto"
    python train_federated.py \
        --model_name "$MODEL" \
        --aggregation ditto \
        --ditto_lambda 0.1 \
        --num_rounds $ROUNDS \
        --seed $SEED \
        --experiment_name "ditto_s${SEED}" \
        --output_dir "$OUTPUT_DIR" \
        --results_dir "$RESULTS_DIR"

    # B5: pFedMe + LoRA
    echo ">>> B5: pFedMe"
    python train_federated.py \
        --model_name "$MODEL" \
        --aggregation pfedme \
        --pfedme_lambda 15.0 \
        --num_rounds $ROUNDS \
        --seed $SEED \
        --experiment_name "pfedme_s${SEED}" \
        --output_dir "$OUTPUT_DIR" \
        --results_dir "$RESULTS_DIR"

    # B6: q-FFL + LoRA
    echo ">>> B6: q-FFL"
    python train_federated.py \
        --model_name "$MODEL" \
        --aggregation qffl \
        --qffl_q 1.0 \
        --num_rounds $ROUNDS \
        --seed $SEED \
        --experiment_name "qffl_s${SEED}" \
        --output_dir "$OUTPUT_DIR" \
        --results_dir "$RESULTS_DIR"

    # B7: FedAvg + local dual constraints
    echo ">>> B7: FedAvg + Local Dual"
    python train_federated.py \
        --model_name "$MODEL" \
        --aggregation fedavg_dual \
        --num_rounds $ROUNDS \
        --seed $SEED \
        --experiment_name "fedavg_dual_s${SEED}" \
        --output_dir "$OUTPUT_DIR" \
        --results_dir "$RESULTS_DIR"

    # B8: Ditto + local dual constraints
    echo ">>> B8: Ditto + Local Dual"
    python train_federated.py \
        --model_name "$MODEL" \
        --aggregation ditto_dual \
        --ditto_lambda 0.1 \
        --num_rounds $ROUNDS \
        --seed $SEED \
        --experiment_name "ditto_dual_s${SEED}" \
        --output_dir "$OUTPUT_DIR" \
        --results_dir "$RESULTS_DIR"

    # B9: Centralized pooled constrained (oracle upper bound)
    # Simulated by running fedhypca with 1 org that has all data
    echo ">>> B9: Centralized Oracle"
    python train_federated.py \
        --model_name "$MODEL" \
        --aggregation fedhypca \
        --num_orgs 1 \
        --train_size_per_org 72000 \
        --num_rounds $ROUNDS \
        --seed $SEED \
        --experiment_name "centralized_s${SEED}" \
        --output_dir "$OUTPUT_DIR" \
        --results_dir "$RESULTS_DIR"

    # B10: Local-only fine-tuning (no federation)
    echo ">>> B10: Local-Only"
    python train_federated.py \
        --model_name "$MODEL" \
        --aggregation fedhypca \
        --no_federation \
        --num_rounds 1 \
        --local_epochs 9 \
        --seed $SEED \
        --experiment_name "local_only_s${SEED}" \
        --output_dir "$OUTPUT_DIR" \
        --results_dir "$RESULTS_DIR"

done

echo "All baselines completed!"
