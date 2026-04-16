# Experiment Plan: Fed-HyPCA

**Date**: 2026-04-06
**Target**: NeurIPS 2026
**Total Budget**: ~2,500 GPU-hours (A100-80GB)

---

## Claim-Driven Experiment Roadmap

### Core Claims to Validate

| # | Claim | Required Evidence |
|---|-------|-------------------|
| C1 | FedAvg produces post-aggregation policy violations | Show AggViol >> PersViol for FedAvg |
| C2 | Constrained aggregation reduces AggViol with bounded utility loss | AggViol(ours) << AggViol(FedAvg), utility gap < 5% |
| C3 | Strict-minority orgs are protected | Worst-org violation(ours) << worst-org violation(baselines) |
| C4 | Post-aggregation violation is bounded | Empirical verification of theorem bound |
| C5 | Each component contributes | Ablation shows degradation when any component removed |

---

## Benchmark Design

### 6 Organizations

| Org | Domain | Must-Refuse (of K=8) | Gray-Zone Allowed | Strictness |
|-----|--------|---------------------|-------------------|------------|
| O1 | K-12 Tutor | self-harm, medical, sexual, weapons, hate, privacy (6/8) | basic finance | Very strict |
| O2 | Telehealth Triage | self-harm instructions, medical dosing, privacy (3/8) | general wellness | Strict |
| O3 | Consumer Finance | fraud, tax evasion, money laundering, privacy (4/8) | budgeting, credit | Moderate |
| O4 | Enterprise IT | offensive cyber, credential theft, privacy, weapons (4/8) | defensive security | Moderate |
| O5 | Creative Writing | real-world violence, targeted hate, privacy (3/8) | fictional violence, adult fiction | Lenient |
| O6 | Peer Support | actionable self-harm, targeted abuse, privacy (3/8) | emotional support, recovery | Moderate-lenient |

### Data Sources
- Allowed prompts: WildChat, HH-RLHF (helpful split)
- Unsafe/refusal prompts: BeaverTails, HarmBench, StrongREJECT, SORRY-Bench
- All normalized to 8-category taxonomy
- Same prompt relabeled differently across orgs based on policy vectors

### Per-Org Data Split
- Train: 12k prompts
- Val: 1.5k prompts
- Test: 2k prompts
- Jailbreak eval: 500 prompts (HarmBench + StrongREJECT)

---

## Baselines

| # | Method | Source | Why Included |
|---|--------|--------|-------------|
| B1 | FedAvg + LoRA | McMahan et al. 2017 | Standard FL baseline |
| B2 | FedProx + LoRA | Li et al. 2020 | Proximal FL baseline |
| B3 | SCAFFOLD + LoRA | Karimireddy et al. 2020 | Variance-reduced FL |
| B4 | Ditto + LoRA | Li et al. 2021 | Personalized FL baseline |
| B5 | pFedMe + LoRA | T Dinh et al. 2020 | Moreau-envelope personalized FL |
| B6 | q-FFL + LoRA | Li et al. 2020 | Fair FL baseline |
| B7 | FedAvg + local dual constraints | Ours (ablation) | Tests if local constraints alone suffice |
| B8 | Ditto + local dual constraints | Ours (ablation) | Tests personalized FL + local constraints |
| B9 | Centralized pooled constrained | Oracle upper bound | Best possible with all data |
| B10 | Local-only fine-tuning | Privacy lower anchor | No federation |

---

## Metrics

### Primary (Main Table)

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| AggViol | (1/Σ|M_i|) Σ_i Σ_{k∈M_i} [g_{i,k}(θ)]_+ | Shared model violation |
| PersViol | (1/N) Σ_i (1/|M_i|) Σ_{k∈M_i} [g_{i,k}(u_i)]_+ | Personalized model violation |
| WorstViol | max_i max_{k∈M_i} [g_{i,k}(u_i)]_+ | Worst-org worst-category violation |
| RefRecall_i | fraction of unsafe prompts correctly refused | Per-org refusal recall |
| OverRefusal_i | fraction of benign prompts incorrectly refused | Per-org false refusal rate |
| UtilityNLL | avg NLL on allowed prompts | Helpfulness |
| ASR | attack success rate on jailbreak prompts | Robustness |
| ConsensusDist | (1/N) Σ_i ||u_i - θ||² | Federation cohesion |

### Secondary (Appendix)
- Per-org constraint satisfaction rate
- Std-dev across orgs
- Pareto frontier: UtilityNLL vs AggViol
- Communication overhead (bytes per round)
- Wall-clock time per round

---

## Ablation Matrix

| ID | Variant | Removed Component | Expected Failure |
|----|---------|-------------------|-----------------|
| A1 | FedAvg server | Constrained aggregation QP | High AggViol |
| A2 | Scalar violation weights only | Linearized Jacobian terms | Better than FedAvg, worse than full |
| A3 | No slack variables | Soft feasibility | Infeasible/unstable server |
| A4 | Fixed penalty coefficients | Adaptive dual updates | Under/over-enforcement |
| A5 | No proximal term | Consensus link | Personalized drift |
| A6 | Shared-only (u_i = θ) | Personalization | Poor per-org fit |
| A7 | Local-only (no θ) | Federation | Weak cross-org utility |
| A8 | No over-refusal constraint | Utility preservation | Excess refusals |
| A9 | No refusal head aux loss | Optimization aid | Slower convergence |
| A10 | Single "unsafe" bit | Structured policy | Poor gray-zone handling |
| A11 | Text-only policy grounding | Structured vector | Higher variance |
| A12 | Frozen λ after warm-start | Dynamic duals | Weak adaptation |

---

## 10-Week Execution Roadmap

### Week 1: Infrastructure + Toy Benchmark
- [ ] Finalize 8-category taxonomy with clear definitions
- [ ] Build 2-org toy benchmark (O4 IT + O5 Creative, 2 categories each)
- [ ] Set up LoRA training scaffold on Qwen-2.5-3B-Instruct
- [ ] Implement refusal head (linear probe on [CLS]/last token)
- [ ] Build metric pipeline (all 8 primary metrics)
- **Deliverable**: Training loop runs, metrics compute correctly

### Week 2: Local Constrained Training
- [ ] Implement differentiable constraint surrogates g_{i,k}^ref and g_i^ben
- [ ] Implement local primal-dual update (Algorithm 1)
- [ ] Validate on small classifier before LLM: constraints become feasible
- [ ] Tune τ, η_λ, η_ν on toy benchmark
- **Deliverable**: Local models satisfy constraints on 2-org toy

### Week 3: Server Constrained Aggregation
- [ ] Implement server QP solver (linearized constrained consensus)
- [ ] Test on synthetic quadratic toy: show FedAvg infeasible, ours bounded
- [ ] Implement FedAvg baseline for comparison
- [ ] Verify theorem bound empirically on toy
- **Deliverable**: Core figure — "local feasible, FedAvg infeasible, ours bounded"

### Week 4: MVP 2-Org LLM Experiment
- [ ] Run full pipeline on 2-org toy with Qwen-2.5-3B
- [ ] Compare: FedAvg vs FedAvg+local-dual vs full method
- [ ] Lock the core result: constrained aggregation reduces AggViol
- [ ] Debug any training instabilities
- **Deliverable**: MVP result confirming core claim

### Week 5: 6-Org Benchmark Construction
- [ ] Build full 6-org dataset from WildChat, HH-RLHF, BeaverTails, HarmBench
- [ ] Implement org policy vectors and category mapping
- [ ] Create train/val/test splits per org
- [ ] Set up all 10 baseline training scripts
- **Deliverable**: Complete benchmark ready for experiments

### Week 6: Baseline Runs
- [ ] Run B1-B6 (FedAvg, FedProx, SCAFFOLD, Ditto, pFedMe, q-FFL) on 3B, 3 seeds
- [ ] Run B7-B8 (FedAvg+local-dual, Ditto+local-dual)
- [ ] Run B9-B10 (centralized oracle, local-only)
- [ ] Collect all metrics
- **Deliverable**: Complete baseline results table

### Week 7: Full Method + Ablations
- [ ] Run full Fed-HyPCA on 3B, 3 seeds
- [ ] Run ablations A1-A12
- [ ] Sweep key hyperparameters: ρ, β, β_b, η_λ, α_{i,k}, β_i
- [ ] Generate Pareto plots and training curves
- **Deliverable**: Main results + ablation table

### Week 8: Robustness + Safety Evaluation
- [ ] Run jailbreak evaluation (HarmBench, StrongREJECT, SORRY-Bench)
- [ ] Test with 1 malicious client (preference poisoning, 5% corrupted data)
- [ ] 200-example human audit for judge calibration
- [ ] Compute ASR across all methods
- **Deliverable**: Robustness table + safety analysis

### Week 9: Scale-Up + Theory
- [ ] Run top 3 methods + full method on Llama-3.1-8B for confirmation
- [ ] Finalize theorem proof and write theory section
- [ ] Failure analysis: when does the method NOT help?
- [ ] Generate all figures
- **Deliverable**: 7B confirmation + complete theory section

### Week 10: Paper Writing + Polish
- [ ] Write full paper draft (9 pages)
- [ ] Finalize appendix (proofs, additional results, hyperparameters)
- [ ] Internal review and revision
- [ ] Reserve compute for any missing cells or negative results
- **Deliverable**: Submission-ready draft

---

## Compute Budget

| Phase | GPU-Hours (A100-80GB) |
|-------|----------------------|
| Weeks 1-4: Infrastructure + MVP | 200 |
| Week 5: Benchmark construction | 50 |
| Week 6: Baseline runs (10 methods × 3 seeds × 3B) | 700 |
| Week 7: Full method + 12 ablations × 3 seeds × 3B | 900 |
| Week 8: Robustness eval + judge runs | 250 |
| Week 9: 7B confirmation (4 methods × 3 seeds) | 350 |
| Contingency | 300 |
| **Total** | **~2,750** |

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| QP solver unstable | Solve blockwise per adapter layer; fallback to penalized unconstrained + 5 projected steps |
| Refusal head noisy | Larger unsafe minibatches; EMA-smoothed constraint estimates |
| Utility collapses | Lower dual step size; increase benign cap weight; warm-start from helpful-only adapters |
| Text policy grounding weak | Keep structured vectors in main paper; text grounding → appendix |
| 6-org benchmark too small | Scale with policy perturbations and gray-zone relabeling, not model size |
| Automated judges noisy | Classifier-based refusal metrics as primary; human-audited subset as secondary |
| Post-aggregation bound too loose | Report empirical gap alongside theoretical bound; discuss tightness |
| Baselines too strong | This is actually good — shows the problem is hard and our method still helps |

---

## Experiment Tracker

| Experiment | Status | Claim | Priority |
|-----------|--------|-------|----------|
| 2-org toy (synthetic quadratic) | Pending | C1, C4 | P0 |
| 2-org MVP (Qwen-3B) | Pending | C1, C2 | P0 |
| 6-org baselines (Qwen-3B) | Pending | C1, C3 | P0 |
| 6-org full method (Qwen-3B) | Pending | C2, C3 | P0 |
| 6-org ablations (Qwen-3B) | Pending | C5 | P1 |
| Robustness eval | Pending | C2 | P1 |
| 7B confirmation | Pending | C2, C3 | P2 |
| Human audit (200 examples) | Pending | Calibration | P2 |
| Text policy grounding ablation | Pending | Supporting | P3 |
