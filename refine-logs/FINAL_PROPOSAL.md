# Fed-HyPCA: Refined Proposal

**Version**: Post-Review Refinement v2
**Date**: 2026-04-07
**Target**: NeurIPS 2026
**Review Score**: 5/10 → Target 7+

---

## Reframed Contribution

**Personalized Constrained Consensus for Federated Safety Alignment**: A server-side constrained aggregation algorithm that respects heterogeneous organizational safety policies through violation-aware consensus optimization.

The work focuses on the **constrained aggregation** problem, not full LLM alignment. The refusal head serves as a differentiable safety proxy for efficient federated optimization.

---

## Problem Anchor (FROZEN)

**Learn one shared model and one adapter per organization so that each client satisfies its own refusal/compliance policy while remaining close enough to a transferable global consensus — and prove that the aggregated model does not silently violate local constraints.**

---

## Dominant Contribution

**Violation-aware constrained aggregation**: a server-side linearized constrained consensus step that explicitly penalizes directions predicted to reintroduce policy violations after aggregation, with a post-aggregation violation bound.

This is NOT "FedAvg + safety extras." The safety signal changes the geometry of aggregation itself.

---

## Method Overview

### Setup
- N organizations, frozen base LLM M₀
- Shared server parameter: θ ∈ ℝ^d (LoRA backbone)
- Client personalized parameter: u_i ∈ ℝ^d (LoRA adapter)
- Client model: M₀ ⊕ u_i
- Org policy: structured vector m_i ∈ {0,1}^K, K=8 safety categories
- Refusal score: r_{u_i}(x) ∈ (0,1), lightweight linear probe on prompt representation

### K=8 Policy Categories
1. Self-harm
2. Medical treatment advice
3. Financial/legal fraud
4. Cyber misuse
5. Weapons/violent wrongdoing
6. Sexual content
7. Hate/harassment
8. Privacy/doxxing

### Optimization Problem

```
min_{θ, {u_i}} Σ_i p_i [ L_i^allow(u_i) + γ L_i^aux(u_i) + ρ/2 ||u_i - θ||² ]

s.t.  g_{i,k}^ref(u_i) ≤ 0,  ∀i, k ∈ M_i    (must-refuse constraints)
      g_i^ben(u_i) ≤ 0,      ∀i                (over-refusal cap)
```

Where:
- L_i^allow: NLL on allowed prompts (utility)
- L_i^aux: BCE loss for refusal head stability
- ρ/2 ||u_i - θ||²: proximal consensus term

### Constraint Surrogates (Differentiable)

**Must-refuse** (category k for org i):
```
g_{i,k}^ref(u_i) = E_{x~U_{i,k}} [ τ log(1 + exp((α_{i,k} - r_{u_i}(x))/τ)) ] - ε_{i,k}
```
Interpretation: smooth penalty when refusal probability drops below floor α_{i,k}.

**Over-refusal cap** (benign prompts):
```
g_i^ben(u_i) = E_{x~B_i} [ τ log(1 + exp((r_{u_i}(x) - β_i)/τ)) ] - ε_i^ben
```
Interpretation: smooth penalty when refusal probability on benign prompts exceeds cap β_i.

### Client Local Update (Primal-Dual)

Local Lagrangian:
```
J_i(u_i, λ_i, ν_i; θ) = L_i^allow(u_i) + γ L_i^aux(u_i) + ρ/2 ||u_i - θ||²
                        + Σ_{k∈M_i} λ_{i,k} g_{i,k}^ref(u_i)
                        + ν_i g_i^ben(u_i)
                        - μ/2 (||λ_i||² + ν_i²)
```

For s = 0, ..., S-1 local steps:
```
u_i^{t,s+1} = u_i^{t,s} - η_u ∇_{u_i} J_i(u_i^{t,s}, λ_i^{t,s}, ν_i^{t,s}; θ^t)
λ_{i,k}^{t,s+1} = [λ_{i,k}^{t,s} + η_λ g_{i,k}^ref(u_i^{t,s+1})]_+
ν_i^{t,s+1} = [ν_i^{t,s} + η_ν g_i^ben(u_i^{t,s+1})]_+
```

Client sends to server: u_i^{t+1}, ĝ_{i,k}^{t+1}, Ĵ_{i,k}^{t+1} (constraint values + Jacobians)

### Server Aggregation (MAIN CONTRIBUTION)

Instead of FedAvg (θ^{t+1} = Σ_i p_i u_i^{t+1}), solve:

```
(θ^{t+1}, ξ, ζ) = argmin_{θ, ξ≥0, ζ≥0}
    Σ_i p_i ||θ - u_i^{t+1}||² + β Σ_i Σ_{k∈M_i} ξ_{i,k} + β_b Σ_i ζ_i

s.t.  ĝ_{i,k}^{t+1} + ⟨Ĵ_{i,k}^{t+1}, θ - u_i^{t+1}⟩ ≤ ξ_{i,k},  ∀i,k
      ĝ_{i,ben}^{t+1} + ⟨Ĵ_{i,ben}^{t+1}, θ - u_i^{t+1}⟩ ≤ ζ_i,    ∀i
```

**Why this matters**: It explicitly penalizes aggregation directions predicted to reintroduce policy violations. The linearized constraint Jacobians provide first-order correction beyond naive averaging.

### Theorem Target (Realistic Bound)

Assume each g_{i,k} is L_g-smooth. If the server solves the subproblem above:

```
g_{i,k}(θ^{t+1}) ≤ ξ_{i,k}^{t+1} + (L_g/2) ||θ^{t+1} - u_i^{t+1}||²
```

Post-aggregation violation is bounded by: (1) chosen slack, and (2) consensus distance.

Secondary appendix theorem: O(T^{-1/2}) bound on average KKT residual.

---

## Policy Representation (Supporting, Simplified)

**Main paper**: Structured policy vector m_i ∈ {0,1}^K with continuous thresholds α_{i,k}, β_i.

**Appendix/ablation**: Text policy grounding — map org's written policy doc into (m_i, α_i, β_i) via external parser, then run same method.

---

## What Reviewers Remember

"FedAvg can turn locally feasible personalized safety policies into an infeasible shared model. Our server-side constrained consensus step fixes that with a principled aggregation rule and a post-aggregation violation bound."

---

## Paper Structure (9 pages + appendix)

| Section | Pages | Content |
|---------|-------|---------|
| 1. Introduction | 0.75 | Core claim, failure mode illustration |
| 2. Problem Setup | 1.00 | Personalized constrained consensus formulation |
| 3. Method | 2.00 | Local primal-dual + server constrained aggregation |
| 4. Theory | 1.00 | Feasibility-transfer bound + optimization result |
| 5. Benchmark | 1.00 | 6 orgs, 8 categories, data sources |
| 6. Main Results | 1.50 | Comparison tables + Pareto plots |
| 7. Ablations | 1.00 | 12-row ablation matrix + diagnostics |
| 8. Limitations | 0.50 | Scope of claims, ethics |
| 9. Conclusion | 0.25 | Summary |

### Figure Plan
- Fig 1: Geometry — two locally feasible points whose average is infeasible
- Fig 2: Method diagram — local primal-dual + constrained consensus
- Fig 3: Org-policy matrix — 6 orgs × 8 categories with conflicts
- Fig 4: Main Pareto plot — utility vs aggregate violation
- Fig 5: Training curves — AggViol, ConsensusDist, over-refusal across rounds
- Table 1: Main comparison results
- Table 2: Ablation matrix
- Table 3: Robustness evaluation

---

## Differentiation from Closest Competitors

| Paper | What They Do | What We Add |
|-------|-------------|-------------|
| FIRM (2511.16992) | Federated multi-obj alignment via regularized MGDA | Per-org hard constraints via dualization + constrained aggregation |
| FedPDPO (2603.19741) | Federated personalized DPO with adapter heads | Policy-conditioned constraints + violation-aware aggregation |
| FedBiscuit (ICLR'25) | Federated RLHF with binary selectors | Explicit safety constraints + constrained consensus |
| One-Shot Dualization (NeurIPS'24) | Centralized constrained alignment | Federated extension with aggregation-aware feasibility |
| HIPO (2603.16152) | Primal-dual constrained alignment (centralized) | Federated + per-org heterogeneous constraints |
| Panacea (NeurIPS'24) | SVD-based policy vectors (centralized) | Federated + constrained + violation-aware aggregation |
