# Auto Review Log: Fed-HyPCA

**Started**: 2026-04-07
**Target**: NeurIPS 2026

---

## Round 1 (2026-04-07)

### Assessment
- Score: 4/10
- Verdict: Not ready

### Actions Taken
1. Reframed title to "Personalized Constrained Consensus"
2. Fixed server constraint estimation over full validation
3. Fixed violation metrics to include OverRefusal
4. Implemented A1, A6, A10 ablations

---

## Round 2 (2026-04-07)

### Assessment
- Score: 3/10
- Verdict: Not ready

### Actions Taken
1. Fixed Jacobian computation (was zero before)
2. Fixed theorem bound to use linearized slack
3. Fixed A6 ablation (skips local training)

---

## Round 3 (2026-04-07)

### Assessment
- Score: 4/10 (from GPT-5.4)
- Verdict: Not ready - need Pareto curves and lower OverRefusal

### Fixes Applied

1. **Hyperparameters**:
   - `beta_b`: 1.0 → 50.0 (server over-refusal penalty)
   - `eta_nu`: 0.01 → 0.1 (dual step size)
   - `nu` initialization: 0 → 0.5

2. **Code fixes**:
   - Synced CLI defaults with config defaults (train_federated.py)
   - Updated run_fedhypca.sh with new defaults
   - Fixed WorstViol to include OverRefusal

3. **Investigated**:
   - Stratified Jacobian fix caused regression - reverted
   - Ran beta_b sweep: 10, 25, 50, 100, 200

### Pareto Sweep Results (beta_b)

| beta_b | RefRecall | OverRefusal | AggViol | PersViol |
|--------|-----------|-------------|---------|----------|
| 10 | 0.257 | 0.090 | 0.509 | 0.470 |
| 25 | 0.238 | 0.062 | 0.512 | 0.471 |
| 50 | 0.000 | 0.000 | 0.578 | 0.563 |
| 100 | 0.000 | 0.000 | 0.578 | 0.563 |
| 200 | 1.000 | 0.988 | 0.230 | 0.246 |
| FedAvg | 0.224 | 0.039 | 0.511 | 0.470 |

### Key Findings

1. **U-shaped curve**: beta_b shows non-monotonic behavior
   - Low (10-25): Moderate RefRecall, low OverRefusal
   - Medium (50-100): Model refuses nothing (under-fitting)
   - High (200): Model refuses everything (over-fitting)

2. **Best operating point**: beta_b=25 gives:
   - RefRecall: 24% (slightly better than FedAvg 22%)
   - OverRefusal: 6% (slightly higher than FedAvg 4%)
   - Marginal improvement over FedAvg

3. **Stratified Jacobian regression**: The "fix" to stratify Jacobians caused the model to learn nothing. Reverted to non-stratified version with 4 batches.

4. **Remaining issues**:
   - Improvement over FedAvg is marginal
   - Need to understand why high beta_b causes refusal collapse again
   - Need more training rounds for convergence

### Status
- Refusal collapse resolved at moderate beta_b
- Method shows marginal improvements over FedAvg
- GPT-5.4 score: 4/10 (unchanged)
- **Critical finding**: Marginal improvement is NOT sufficient for a method paper at top venue

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Score**: 4/10 for a top venue.

**Verdict**: No.

I verified the claimed fixes: CLI/config/script defaults are now aligned. `WorstViol` now includes over-refusal. Those are real improvements.

But the empirical signal is still too weak. On the sweep, the only acceptable low-overrefusal point is basically a marginal improvement over FedAvg. That is not enough for a method paper at NeurIPS/ICML level.

**Remaining Weaknesses**:
1. The main empirical win is still marginal. Need 6 orgs, 50 rounds, 3 seeds.
2. The U-shaped sweep suggests instability or score-calibration failure. Need score histograms.
3. The Jacobian pass is still not guaranteed-coverage.
4. A6 and A10 are still not valid end-to-end ablations.
5. `UtilityNLL` is still a prompt-reconstruction proxy.

**Next Steps**:
1. Diagnose the U-shape before spending large compute.
2. Choose an operating rule on validation, not by eyeballing one `beta_b`.
3. Then run the full 6-org, 50-round, multi-seed experiment.
4. If the gains stay this small, this is not a top-venue method paper; it becomes a narrower diagnostic/analysis paper instead.

</details>

---

## Summary: Round 3 Complete

### What Was Fixed
1. Refusal collapse resolved (beta_b tuning)
2. CLI/config/script defaults synchronized
3. WorstViol includes OverRefusal
4. Pareto sweep conducted

### Critical Issue
**Method shows only marginal improvements over FedAvg in sanity checks.**

At beta_b=25 (best low-overrefusal point):
- RefRecall: 24% vs FedAvg 22% (+9%)
- OverRefusal: 6% vs FedAvg 4% (+50%)

This is **insufficient** for a method paper at top venue.

### Remaining Work (High Effort)
1. Diagnose U-shaped beta_b behavior
2. Implement score histograms and calibration analysis
3. Run full 6-org, 50-round, 3-seed experiments
4. Fix A6/A10 ablations end-to-end
5. If gains remain marginal, pivot to analysis paper

### Decision Point
- Continue with full experiments (~100 GPU-hours)?
- Pivot to analysis/diagnostic paper?
- Reframe method for workshop/other venue?
