# Fed-HyPCA Project

Federated Hybrid Policy-Conditioned Alignment — personalized constrained consensus for federated LLM alignment under heterogeneous organizational policies. Targeting NeurIPS 2026.

## Remote Server

- SSH: `ssh root@10.132.166.5`
- Password: `qiqiadmin123`
- GPU: AutoDL (RTX 5880 ada / RTX 6000, port changes on restart)
- Conda: `eval "$(/root/miniconda3/bin/conda shell.bash hook)" && conda activate qiqi`
- Code: `/home/qiqi/Fed-Hy`
- Models: `/home/qiqi/models` (qwen0.5b/, qwen3.5-4b/)
- Sync: `rsync -avz -e "ssh" local_file root@10.132.166.5:/home/qiqi/Fed-Hy/`
- Background jobs: `nohup bash -c '...' > logs/experiment.log 2>&1 &`
- Check progress: `tail -f logs/experiment.log` or `grep -v "\[A" logs/experiment.log | tail -20`

## Pipeline Status

- **Stage**: validation (verifying fixes)
- **Model**: Qwen3.5-4B-base (4-bit quantized, LoRA rank 16)
- **Current**: quick test running — 5 rounds Fed-HyPCA + 5 rounds FedAvg
- **Bug tracker**: `refine-logs/BUG-FIX.md`

## Project Structure

```
Fed-Hy/
├── train_federated.py              # Training entry point (all methods + ablations)
├── evaluate.py                     # Standalone evaluation
├── configs/default.py              # All configs: model, training, data, eval, ablation flags
├── src/
│   ├── models/
│   │   ├── lora_model.py           # FedHyPCAModel: base LLM + LoRA + RefusalHead
│   │   └── refusal_head.py         # Linear probe for refusal scoring
│   ├── data/
│   │   ├── taxonomy.py             # 8-category safety taxonomy + 6 org policies
│   │   └── dataset.py              # BeaverTails + HH-RLHF loading, per-org splitting, pickle cache
│   ├── constraints/surrogates.py   # Differentiable constraint surrogates + DualVariables
│   ├── federated/
│   │   ├── aggregation.py          # FedAvg, q-FFL, SCAFFOLD, Fed-HyPCA constrained aggregation
│   │   ├── client.py               # Client local primal-dual training + accumulated dual updates
│   │   └── server.py               # Server orchestration + zero-Jacobian auto-detection
│   ├── evaluation/metrics.py       # 8 primary metrics + result saving
│   └── utils/seed.py               # Reproducibility
├── scripts/
│   ├── run_validation.sh           # Quick validation (Fed-HyPCA + FedAvg comparison)
│   ├── run_fedhypca.sh             # Full method x3 seeds
│   ├── run_baselines.sh            # All baselines x3 seeds
│   ├── run_ablations.sh            # 12 ablations x3 seeds
│   ├── run_robustness.sh           # Poisoning experiments
│   ├── toy_quadratic.py            # CPU-only synthetic experiment (validates core claim)
│   └── aggregate_results.py        # Aggregate results into comparison tables
└── refine-logs/
    ├── FINAL_PROPOSAL.md           # Refined method with math + paper structure
    ├── EXPERIMENT_PLAN.md          # Full experiment plan + roadmap
    ├── EXPERIMENT_TRACKER.md       # Experiment status tracker
    └── BUG-FIX.md                  # Bug tracker with fix history
```

## Key Commands

```bash
# Fed-HyPCA (main method, current defaults)
python train_federated.py --aggregation fedhypca --num_rounds 50 --seed 42

# FedAvg baseline
python train_federated.py --aggregation fedavg --num_rounds 50 --seed 42

# Quick validation (5 rounds, 50 steps/client, eval every round)
python -u train_federated.py --aggregation fedhypca --num_rounds 5 \
    --local_steps_per_epoch 50 --batch_size 1 --eval_every 1

# CPU-only core-claim validation
python scripts/toy_quadratic.py
```

## Key Parameters (current defaults)

| Parameter | Value | Notes |
|-----------|-------|-------|
| rho | 0.01 | Proximal consensus weight (low for personalization) |
| eta_lambda | 0.05 | Dual step size for refusal constraints |
| eta_nu | 0.1 | Dual step size for over-refusal |
| scalar_reweight_beta | 5.0 | Exponential violation reweighting (scalar path) |
| dual_update_interval | 10 | Accumulate N steps before dual update |
| beta_b | 50.0 | Over-refusal slack penalty |
| tau | 0.1 | Softplus temperature |

## Aggregation Methods

`fedhypca`, `fedavg`, `fedprox`, `scaffold`, `ditto`, `pfedme`, `qffl`, `fedavg_dual`, `ditto_dual`

## Ablation Flags

`--no_constrained_aggregation` (A1), `--no_jacobian_correction` (A2), `--no_slack_variables` (A3), `--no_adaptive_duals` (A4), `--no_proximal_term` (A5), `--no_personalization` (A6), `--no_federation` (A7), `--no_overrefusal_constraint` (A8), `--no_refusal_head_aux` (A9), `--no_structured_policy` (A10), `--freeze_duals_after N` (A12)

## Known Limitations

- Jacobian computation disabled for Qwen3.5-4B (OOM), using scalar-weight-only aggregation (see BUG-FIX.md B1)
- 6 clients run sequentially (no parallel client training)
- Use `python -u` to avoid stdout buffering issues with `tee`
