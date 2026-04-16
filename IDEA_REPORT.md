# Idea Discovery Report

**Direction**: Federated Hybrid Policy-Conditioned Alignment (Fed-HyPCA)
**Date**: 2026-04-06
**Pipeline**: research-lit → idea-creator → novelty-check → research-review → research-refine-pipeline

---

## Executive Summary

From a survey of ~40 papers (2023-2026), two rounds of GPT-5.4 idea generation (11 standalone + 3 composite ideas), deep novelty verification (30+ targeted searches), and two rounds of brutal external review (score 5/10 → actionable path to 7+), we converge on a refined proposal:

**Fed-HyPCA: Personalized Constrained Consensus for Federated LLM Alignment** — learn one shared model and one adapter per organization so that each client satisfies its own refusal/compliance policy while remaining close to a transferable global consensus. The dominant contribution is a **violation-aware constrained aggregation** rule with a post-aggregation feasibility bound.

All three novelty claims are **confirmed novel** as of April 2026. The field is heating up fast (4+ new federated alignment papers in 2025-2026), but no one has claimed this specific intersection.

---

## Literature Landscape

### Research Threads (40+ papers surveyed)

| Thread | Key Papers | Status |
|--------|-----------|--------|
| Federated LLM Alignment | FedBiscuit (ICLR'25), PluralLLM, FedPDPO (2026), KNEXA-FL (AAAI'26), FIRM (2025), FedMOA (2026) | Emerging — no constraints, no policy conditioning |
| Safety in Federated FT | Ye et al. (ICLR'25), Li et al. (2024), Guo et al. (EMNLP'25) | Early — attacks shown, defenses weak |
| Policy-Conditioned Generation | Panacea (NeurIPS'24), DPA, MOSAIC (2026), RepE, RC-GRPO | Active — centralized only |
| Constrained Alignment | One-Shot Dualization (NeurIPS'24), Safe RLHF (ICLR'24), CS-RLHF, C-DPO, HIPO (2026) | Maturing — centralized only |
| Robust/Fair Aggregation | GRPO (NeurIPS'24), FairFed (AAAI'23), FedFACT (NeurIPS'25), MaxMin-RLHF | Mature for classification, not alignment |
| Personalized FL | FedMCP, FedAMoLE, FDLoRA, FedMCP | Active — adapter-level, no policy semantics |
| Alignment Poisoning | Best-of-Venom, RLHFPoison, Nika et al. (2025), Hammoud et al. (2024) | Growing — theoretical bounds emerging |

### Confirmed Gaps
1. **No** policy-conditioned alignment + federated learning
2. **No** safety constraints via dualization in federated preference optimization
3. **No** robust aggregation for policy vectors in alignment settings
4. Federated DPO/RLHF exists but lacks explicit constraints and policy conditioning
5. Safety degrades trivially under fine-tuning (3% params suffice) — no federated defense
6. Preference poisoning needs only 1-5% corrupted data — federated setting amplifies
7. Model merging propagates misalignment from one bad model

---

## Standalone Ideas (Round 1)

### Safe Bets

| # | Idea | Feasibility | Novelty | Impact | Risk | Score |
|---|------|:-----------:|:-------:|:------:|:----:|------:|
| 1 | FedPanacea-Dual: Policy-Conditioned Constrained Federated DPO | 9 | 8 | 8 | 4 | 144.0 |
| 2 | SovereignSplit: Shared Helpfulness + Distilled Policy Modules | 8 | 7 | 7 | 4 | 98.0 |
| 3 | Robust Policy Aggregation for Federated Alignment | 9 | 8 | 8 | 5 | 115.2 |
| 4 | Minimax Policy FL: Group-Robust Federated Preference Optimization | 8 | 8 | 8 | 5 | 102.4 |

### Ambitious But Doable

| # | Idea | Feasibility | Novelty | Impact | Risk | Score |
|---|------|:-----------:|:-------:|:------:|:----:|------:|
| 5 | DualEnvelope-FL: Upper-Envelope Aggregation of Safety Multipliers | 7 | 9 | 9 | 6 | 94.5 |
| 6 | PolicyHull-FL: Federated Distillation of Monotone Policy Hull | 6 | 9 | 9 | 7 | 69.4 |
| 7 | PoisonShield-FL: Policy-Consistency Defense Against Preference Poisoning | 8 | 9 | 9 | 6 | 108.0 |
| 8 | FedCorrect: Correction-Based Federated Alignment | 7 | 8 | 8 | 6 | 74.7 |

### Moonshots

| # | Idea | Feasibility | Novelty | Impact | Risk | Score |
|---|------|:-----------:|:-------:|:------:|:----:|------:|
| 9 | PolicyLattice: Monotone Sovereignty Constraints | 5 | 10 | 9 | 8 | 56.3 |
| 10 | FedConstitutional Critic: Federated Policy-Conditioned Judge | 6 | 8 | 8 | 7 | 54.9 |
| 11 | HyperLoRA-Sov: Federated Hypernetworks for Policy-Programmable Alignment | 5 | 10 | 10 | 9 | 55.6 |

---

## Novelty Verification (Phase 3)

### Per-Claim Assessment

| Claim | Verdict | Closest Competitor | Differentiation |
|-------|---------|-------------------|-----------------|
| First policy-conditioned alignment + FL | **NOVEL** | FedPDPO (adapter heads, not policy vectors); FedBiscuit (binary selectors) | Panacea-style continuous policy vectors + text grounding in federated setting |
| First per-org constrained dualization in federated DPO | **NOVEL** | FIRM (regularized MGDA, not dualization); HIPO (primal-dual but centralized); C-DPO (centralized) | Per-organization heterogeneous constraints + federated aggregation-aware feasibility |
| First violation-aware aggregation for policy vectors | **NOVEL** | FedMOA (accuracy-based weighting); Srewa et al. (reward-based) | Constraint-violation-driven weighting on policy vector objects |

### Required Defensive Citations
- **FIRM** (arXiv:2511.16992) — closest overall competitor
- **FedPDPO** (arXiv:2603.19741) — closest on personalized federated DPO
- **HIPO** (arXiv:2603.16152) — closest on primal-dual constrained alignment
- **Srewa et al.** (arXiv:2512.08786) — closest on adaptive federated alignment aggregation
- **FedMOA** (arXiv:2602.00453) — closest on federated GRPO

### Positioning
Frame as intersection of three threads never combined:
- **Thread A**: Policy-conditioned alignment (Panacea, MOSAIC) — centralized only
- **Thread B**: Constrained safe alignment (Safe RLHF, C-DPO, HIPO) — centralized only
- **Thread C**: Federated preference optimization (FedBiscuit, FedPDPO, FIRM) — no policy conditioning, no constraints

---

## External Review (Phase 4)

### GPT-5.4 Review: Score 5/10 (Below Accept)

**Strengths:**
- Problem is real and underexplored
- "Average policy collapse" framing is sharp
- Planned ablations are mostly right
- Protecting minority strict organizations is publishable

**Critical Weaknesses:**
1. Kitchen-sink: 4 layers too many for one paper
2. Post-aggregation feasibility gap: local feasibility ≠ global after averaging
3. Overclaiming on guarantees
4. Evaluation plan underpowered relative to claims

**Path to 7+ (Borderline Accept):**
1. Reframe as **personalized constrained consensus** — each client deploys u_i, not averaged model
2. Pick **ONE dominant contribution**: Layer 3 (violation-aware constrained aggregation)
3. Make math explicit and modest: precise objective, defined surrogates, realistic bound
4. Cut Layer 4 entirely, simplify Layer 1
5. Report results BEFORE and AFTER any local gateway

**Recommended Framing:**
"FedAvg can turn locally feasible personalized safety policies into an infeasible shared model. Our server-side constrained consensus step fixes that with a principled aggregation rule and a post-aggregation violation bound."

---

## 🏆 Refined Proposal: Fed-HyPCA-Core (RECOMMENDED)

### Problem Anchor (FROZEN)
Learn one shared model and one adapter per organization so that each client satisfies its own refusal/compliance policy while remaining close enough to a transferable global consensus.

### Dominant Contribution
**Violation-aware constrained aggregation**: server-side linearized constrained consensus step that explicitly penalizes directions predicted to reintroduce policy violations after aggregation.

### Theorem Target
Post-aggregation violation bounded by: (1) chosen slack + (2) L_g/2 × consensus distance².

### Key Differentiators from Competitors

| Paper | What They Do | What We Add |
|-------|-------------|-------------|
| FIRM | Federated multi-obj via regularized MGDA | Per-org hard constraints + constrained aggregation |
| FedPDPO | Federated personalized DPO with adapter heads | Policy-conditioned constraints + violation-aware agg |
| FedBiscuit | Federated RLHF with binary selectors | Explicit safety constraints + constrained consensus |
| One-Shot Dualization | Centralized constrained alignment | Federated extension with aggregation-aware feasibility |
| HIPO | Primal-dual constrained alignment (centralized) | Federated + per-org heterogeneous constraints |

---

## Eliminated Ideas

| Idea | Reason |
|------|--------|
| PolicyLattice (#9) | Monotone partial order too subjective; feasibility too low |
| HyperLoRA-Sov (#11) | Generalizing from few policies too unstable for first paper |
| FedConstitutional Critic (#10) | Training critic + actor exceeds 10-12 week scope |
| SovereignSplit (#2) | Subsumed by Composite A; distillation adds complexity without gain |

---

## Refined Proposal & Experiment Plan

- **Proposal**: `refine-logs/FINAL_PROPOSAL.md`
  - Complete method with precise optimization objective
  - Constraint surrogates (differentiable)
  - Server constrained aggregation QP (main contribution)
  - Client local primal-dual update
  - Theorem target with realistic bound
  - Paper structure (9 pages + appendix)
  - Figure plan (5 figures + 3 tables)

- **Experiment plan**: `refine-logs/EXPERIMENT_PLAN.md`
  - 6-org benchmark design with data sources
  - 10 baselines
  - 8 primary metrics with formulas
  - 12-row ablation matrix
  - 10-week execution roadmap
  - ~2,750 GPU-hour budget
  - Risk mitigation for each failure mode

- **Tracker**: `refine-logs/EXPERIMENT_TRACKER.md`

---

## Next Steps

- [ ] Implement infrastructure + toy benchmark (Week 1)
- [ ] Validate local constrained training (Week 2)
- [ ] Implement server constrained aggregation + verify theorem (Week 3)
- [ ] Run MVP 2-org experiment (Week 4)
- [ ] Build 6-org benchmark + run baselines (Weeks 5-6)
- [ ] Full method + ablations (Week 7)
- [ ] Robustness evaluation (Week 8)
- [ ] Scale to 7B + theory writeup (Week 9)
- [ ] Paper writing + polish (Week 10)

Or invoke `/run-experiment` to deploy experiments from the plan, or `/auto-review-loop` to iterate until submission-ready.
