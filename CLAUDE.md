# Fed-HyPCA Project

## Overview
Federated Hybrid Policy-Conditioned Alignment — personalized constrained consensus for federated LLM alignment under heterogeneous organizational policies. Targeting NeurIPS 2026.

## Remote Server
  - SSH: `ssh -p 39599 root@connect.westd.seetacloud.com`
  - GPU: AutoDL GPU instance
  - Activate: `eval "$(/root/miniconda3/bin/conda shell.bash hook)" && conda
  activate base`
  - Code directory: `/root/autodl-tmp/Fed-Hy`
  - code_sync: rsync
  - wandb: false
  - Use `screen` for background jobs: `screen -dmS exp0 bash -c '...'`

## Remote Run Conventions
  - Always SSH into the server, then run commands from `/root/autodl-tmp/Fed-Hy`
  - For sanity-stage verification, prefer the project script: `bash scripts/
  run_sanity.sh`
  - Do not replace `bash scripts/run_sanity.sh` with ad-hoc Python commands
  unless debugging a specific issue
  - For full runs, prefer the project scripts:
    - `bash scripts/run_fedhypca.sh`
    - `bash scripts/run_baselines.sh`
    - `bash scripts/run_ablations.sh`
    - `bash scripts/run_robustness.sh`
  - For targeted debugging, allowed fallback:
    - `python train_federated.py --aggregation fedhypca --num_rounds 50 --seed
  42`
    - `python evaluate.py ...`
  - If running long jobs, launch via `screen` on the remote server
  - Save stdout/stderr with `tee` when not using the provided shell scripts

## Recommended Remote Checks
  - Sanity check first:
    - `cd /root/autodl-tmp/Fed-Hy && bash scripts/run_sanity.sh`
  - CPU-only core-claim validation:
    - `cd /root/autodl-tmp/Fed-Hy && python scripts/toy_quadratic.py`
  - Result aggregation:
    - `cd /root/autodl-tmp/Fed-Hy && python scripts/aggregate_results.py
  --results_dir results/baselines results/main`

## Pipeline Status
  stage: implementation
  idea: "Fed-HyPCA: personalized constrained consensus for federated LLM
  alignment under heterogeneous organizational policies"
  current_branch: main
  baseline: "sanity script + toy quadratic ready"
  training_status: idle
  active_tasks:
  - "none"
  next: "run sanity on remote server, then deploy main/baseline experiments"

## Project Structure
```
Fed-Hy/
├── train_federated.py          # Main training entry point (all methods + ablations)
├── evaluate.py                 # Standalone evaluation script
├── requirements.txt            # Python dependencies
├── configs/
│   └── default.py              # All configs: model, training, data, eval, ablation flags
├── src/
│   ├── models/
│   │   ├── lora_model.py       # FedHyPCAModel: base LLM + LoRA + RefusalHead
│   │   └── refusal_head.py     # Lightweight linear probe for refusal scoring
│   ├── data/
│   │   ├── taxonomy.py         # 8-category safety taxonomy + org policy definitions
│   │   └── dataset.py          # Data loading (BeaverTails, HH-RLHF), per-org splitting
│   ├── constraints/
│   │   └── surrogates.py       # Differentiable constraint surrogates + DualVariables
│   ├── federated/
│   │   ├── aggregation.py      # All aggregation strategies (FedAvg, q-FFL, SCAFFOLD, Fed-HyPCA QP)
│   │   ├── client.py           # Client local primal-dual training
│   │   └── server.py           # Server orchestration + round-level logging
│   ├── evaluation/
│   │   └── metrics.py          # All 8 primary metrics + result saving/printing
│   └── utils/
│       └── seed.py             # Reproducibility
├── scripts/
│   ├── run_sanity.sh           # 2-org toy sanity check (<1hr)
│   ├── run_baselines.sh        # All 10 baselines × 3 seeds
│   ├── run_fedhypca.sh         # Full method × 3 seeds
│   ├── run_ablations.sh        # 12 ablations × 3 seeds
│   ├── run_robustness.sh       # Poisoning experiments
│   ├── toy_quadratic.py        # CPU-only synthetic experiment (validates core claim)
│   └── aggregate_results.py    # Aggregate results into comparison tables
└── refine-logs/
    ├── FINAL_PROPOSAL.md       # Refined method with math + paper structure
    ├── EXPERIMENT_PLAN.md      # Full experiment plan + 10-week roadmap
    └── EXPERIMENT_TRACKER.md   # Experiment status tracker
```

## Quick Start
```bash
pip install -r requirements.txt

# 1. Validate core claim on CPU (no GPU needed)
python scripts/toy_quadratic.py

# 2. Sanity check with LLM (1 GPU, <1hr)
bash scripts/run_sanity.sh

# 3. Full experiments
bash scripts/run_fedhypca.sh      # main method
bash scripts/run_baselines.sh     # all baselines
bash scripts/run_ablations.sh     # ablation study

# 4. Aggregate results
python scripts/aggregate_results.py --results_dir results/baselines results/main
```

## Key Commands
```bash
# Fed-HyPCA (main method)
python train_federated.py --aggregation fedhypca --num_rounds 50 --seed 42

# FedAvg baseline
python train_federated.py --aggregation fedavg --num_rounds 50 --seed 42

# Ablation A1: no constrained aggregation
python train_federated.py --aggregation fedhypca --no_constrained_aggregation --seed 42

# Local-only (no federation)
python train_federated.py --aggregation fedhypca --no_federation --seed 42
```

## Supported Aggregation Methods
`fedhypca`, `fedavg`, `fedprox`, `scaffold`, `ditto`, `pfedme`, `qffl`, `fedavg_dual`, `ditto_dual`

## Ablation Flags
`--no_constrained_aggregation` (A1), `--no_jacobian_correction` (A2), `--no_slack_variables` (A3), `--no_adaptive_duals` (A4), `--no_proximal_term` (A5), `--no_personalization` (A6), `--no_federation` (A7), `--no_overrefusal_constraint` (A8), `--no_refusal_head_aux` (A9), `--no_structured_policy` (A10), `--freeze_duals_after N` (A12)
