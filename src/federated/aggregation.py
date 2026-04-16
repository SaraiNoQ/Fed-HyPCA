"""Server-side aggregation strategies for Fed-HyPCA.

Includes:
- FedAvg (baseline)
- FedProx (proximal, aggregation same as FedAvg)
- Ditto (personalized, aggregation same as FedAvg)
- pFedMe (Moreau envelope personalization)
- q-FFL (fair federated learning)
- SCAFFOLD (variance reduction)
- Fed-HyPCA constrained aggregation (MAIN CONTRIBUTION)
"""

from collections import OrderedDict

import numpy as np
import torch
from scipy.optimize import minimize


def fedavg_aggregate(
    client_states: list[OrderedDict],
    weights: list[float] = None,
) -> OrderedDict:
    """Standard FedAvg: weighted average of client parameters.

    Args:
        client_states: list of client LoRA state dicts
        weights: per-client weights (default: uniform)

    Returns:
        Aggregated state dict
    """
    n = len(client_states)
    if weights is None:
        weights = [1.0 / n] * n

    agg_state = OrderedDict()
    for key in client_states[0]:
        agg_state[key] = sum(
            w * client_states[i][key].float()
            for i, w in enumerate(weights)
        ).to(client_states[0][key].dtype)

    return agg_state


def qffl_aggregate(
    client_states: list[OrderedDict],
    client_losses: list[float],
    q: float = 1.0,
) -> OrderedDict:
    """q-FFL: Fair Federated Learning with reweighted aggregation.

    Upweights clients with higher loss to improve worst-case performance.
    """
    losses = np.array(client_losses)
    # q-FFL weights: L_i^q / sum(L_j^q)
    q_weights = losses ** q
    q_weights = q_weights / q_weights.sum()
    return fedavg_aggregate(client_states, weights=q_weights.tolist())


def scaffold_aggregate(
    client_states: list[OrderedDict],
    client_controls: list[OrderedDict],
    server_control: OrderedDict,
    weights: list[float] = None,
) -> tuple[OrderedDict, OrderedDict]:
    """SCAFFOLD: variance-reduced aggregation with control variates.

    Returns:
        (aggregated_state, updated_server_control)
    """
    n = len(client_states)
    if weights is None:
        weights = [1.0 / n] * n

    # Aggregate parameters (same as FedAvg)
    agg_state = fedavg_aggregate(client_states, weights)

    # Update server control variate
    new_server_control = OrderedDict()
    for key in server_control:
        delta = sum(
            w * (client_controls[i][key].float() - server_control[key].float())
            for i, w in enumerate(weights)
        )
        new_server_control[key] = (server_control[key].float() + delta).to(
            server_control[key].dtype
        )

    return agg_state, new_server_control


def _flatten_state_dict(state: OrderedDict) -> torch.Tensor:
    """Flatten a state dict into a single 1D tensor."""
    return torch.cat([v.float().reshape(-1) for v in state.values()])


def _unflatten_state_dict(
    flat: torch.Tensor,
    template: OrderedDict,
) -> OrderedDict:
    """Unflatten a 1D tensor back into a state dict matching template shapes."""
    result = OrderedDict()
    offset = 0
    for key, val in template.items():
        numel = val.numel()
        result[key] = flat[offset:offset + numel].reshape(val.shape).to(val.dtype)
        offset += numel
    return result


def fedhypca_constrained_aggregate(
    client_states: list[OrderedDict],
    client_constraint_values: list[dict],
    client_constraint_jacobians: list[dict],
    client_overrefusal_values: list[float],
    client_overrefusal_jacobians: list[dict],
    weights: list[float] = None,
    beta: float = 1.0,
    beta_b: float = 1.0,
    use_jacobian: bool = True,
    use_slack: bool = True,
    scalar_beta: float = 5.0,
) -> OrderedDict:
    """Fed-HyPCA constrained aggregation (MAIN CONTRIBUTION).

    Solves the server-side linearized constrained consensus QP:

    min_{θ} Σ_i p_i ||θ - u_i||² + β Σ_{i,k} ξ_{i,k} + β_b Σ_i ζ_i
    s.t.  ĝ_{i,k} + ⟨Ĵ_{i,k}, θ - u_i⟩ ≤ ξ_{i,k}  (refusal constraints)
          ĝ_{i,ben} + ⟨Ĵ_{i,ben}, θ - u_i⟩ ≤ ζ_i    (over-refusal constraints)
          ξ_{i,k} ≥ 0, ζ_i ≥ 0

    In practice, we solve this via the dual or via projected gradient descent
    on the flattened parameter vector.

    Args:
        client_states: list of client LoRA state dicts (u_i)
        client_constraint_values: list of dicts {category_idx: g_value}
        client_constraint_jacobians: list of dicts {category_idx: flat_jacobian_tensor}
        client_overrefusal_values: list of scalar g_ben values
        client_overrefusal_jacobians: list of flat jacobian tensors
        weights: per-client weights p_i
        beta: slack penalty for refusal constraints
        beta_b: slack penalty for over-refusal constraints
        use_jacobian: if False, use scalar violation weights only (ablation A2)
        use_slack: if False, hard constraints without slack (ablation A3)

    Returns:
        Aggregated state dict θ
    """
    n = len(client_states)
    if weights is None:
        weights = [1.0 / n] * n

    template = client_states[0]

    # Flatten all client states
    u_flat = [_flatten_state_dict(s) for s in client_states]
    d = u_flat[0].shape[0]
    device = u_flat[0].device

    # Start from FedAvg as initial point
    theta_init = sum(w * u for w, u in zip(weights, u_flat))

    if not use_jacobian:
        # Scalar violation-aware reweighting (no Jacobian correction)
        # Upweight clients with higher violations so global model moves toward satisfying them
        violation_scores = []
        for i in range(n):
            total_viol = sum(
                max(0.0, v) for v in client_constraint_values[i].values()
            )
            total_viol += max(0.0, client_overrefusal_values[i])
            violation_scores.append(total_viol)

        # Exponential reweighting: clients with more violations get exponentially higher weight
        viol_arr = np.array(violation_scores)
        if viol_arr.max() > 0:
            adaptive_weights = np.array(weights) * np.exp(scalar_beta * viol_arr)
            adaptive_weights = adaptive_weights / adaptive_weights.sum()
        else:
            adaptive_weights = np.array(weights)

        print(f"    Scalar reweighting: violations={viol_arr.round(3)}, "
              f"weights={adaptive_weights.round(4)} (uniform={weights[0]:.4f})")

        return fedavg_aggregate(client_states, weights=adaptive_weights.tolist())

    # Full constrained aggregation via projected gradient descent
    # We solve the QP approximately using L-BFGS-B on the dual

    # Collect all active constraints
    constraints_info = []  # (client_idx, category_idx_or_'ben', g_value, jacobian_flat)

    for i in range(n):
        for k, g_val in client_constraint_values[i].items():
            if k in client_constraint_jacobians[i]:
                jac = client_constraint_jacobians[i][k]
                if isinstance(jac, torch.Tensor):
                    jac_flat = jac.float().to(device)
                else:
                    jac_flat = torch.zeros(d, device=device)
                constraints_info.append((i, k, float(g_val), jac_flat))

        g_ben = client_overrefusal_values[i]
        if i < len(client_overrefusal_jacobians) and client_overrefusal_jacobians[i] is not None:
            jac_ben = client_overrefusal_jacobians[i]
            if isinstance(jac_ben, torch.Tensor):
                jac_ben_flat = jac_ben.float().to(device)
            else:
                jac_ben_flat = torch.zeros(d, device=device)
        else:
            jac_ben_flat = torch.zeros(d, device=device)
        constraints_info.append((i, "ben", float(g_ben), jac_ben_flat))

    if not constraints_info:
        return _unflatten_state_dict(theta_init, template)

    # Projected gradient descent on θ
    theta = theta_init.clone().requires_grad_(False)
    lr_server = 0.01
    n_steps = 20

    for step in range(n_steps):
        # Gradient of consensus objective: 2 Σ_i p_i (θ - u_i)
        grad_consensus = 2.0 * sum(
            w * (theta - u) for w, u in zip(weights, u_flat)
        )

        # Gradient from linearized constraint penalties
        grad_constraint = torch.zeros_like(theta)
        total_slack_penalty = 0.0

        for (ci, ck, g_val, jac_flat) in constraints_info:
            # Linearized constraint: g + <J, θ - u_i>
            delta = theta - u_flat[ci]
            lin_constraint = g_val + torch.dot(jac_flat, delta)

            if use_slack:
                # With slack: penalize max(0, linearized_constraint)
                if lin_constraint.item() > 0:
                    penalty_weight = beta if ck != "ben" else beta_b
                    grad_constraint += penalty_weight * jac_flat
                    total_slack_penalty += penalty_weight * lin_constraint.item()
            else:
                # Without slack: hard penalty (ablation A3)
                penalty_weight = (beta if ck != "ben" else beta_b) * 10.0
                if lin_constraint.item() > 0:
                    grad_constraint += penalty_weight * jac_flat

        # Update
        theta = theta - lr_server * (grad_consensus + grad_constraint)

    return _unflatten_state_dict(theta.detach(), template)


def compute_aggregation_violation_bound(
    theta_state: OrderedDict,
    client_states: list[OrderedDict],
    client_constraint_values: list[dict],
    client_constraint_jacobians: list[dict],
    client_overrefusal_values: list[float],
    client_overrefusal_jacobians: list[dict],
    L_g: float = 1.0,
) -> dict:
    """Compute the post-aggregation violation bound from the theorem.

    g_{i,k}(θ) ≤ ξ_{i,k}(θ) + (L_g/2) ||θ - u_i||²

    where ξ_{i,k}(θ) = max(0, ĝ_{i,k} + ⟨Ĵ_{i,k}, θ - u_i⟩) is the linearized slack.

    Returns dict with per-org, per-category bounds and actual consensus distances.
    """
    theta_flat = _flatten_state_dict(theta_state)
    bounds = {}

    for i, (client_state, constraint_vals, constraint_jacs) in enumerate(
        zip(client_states, client_constraint_values, client_constraint_jacobians)
    ):
        u_flat = _flatten_state_dict(client_state)
        consensus_dist_sq = torch.sum((theta_flat - u_flat) ** 2).item()
        delta = theta_flat - u_flat

        bounds[i] = {
            "consensus_dist_sq": consensus_dist_sq,
            "per_category": {},
        }
        for k, g_val in constraint_vals.items():
            # Compute linearized slack at θ: ξ = max(0, g + <J, θ - u>)
            if constraint_jacs and k in constraint_jacs:
                jac = constraint_jacs[k]
                if isinstance(jac, torch.Tensor):
                    lin_slack = max(0.0, g_val + torch.dot(jac.float().to(theta_flat.device), delta).item())
                else:
                    lin_slack = max(0.0, g_val)
            else:
                lin_slack = max(0.0, g_val)

            bound = lin_slack + (L_g / 2.0) * consensus_dist_sq
            bounds[i]["per_category"][k] = {
                "local_violation": g_val,
                "linearized_slack": lin_slack,
                "bound": bound,
            }

        # Add over-refusal constraint bound
        if i < len(client_overrefusal_values):
            g_ben = client_overrefusal_values[i]
            if i < len(client_overrefusal_jacobians) and client_overrefusal_jacobians[i] is not None:
                jac_ben = client_overrefusal_jacobians[i]
                if isinstance(jac_ben, torch.Tensor):
                    lin_slack_ben = max(0.0, g_ben + torch.dot(jac_ben.float().to(theta_flat.device), delta).item())
                else:
                    lin_slack_ben = max(0.0, g_ben)
            else:
                lin_slack_ben = max(0.0, g_ben)
            bounds[i]["overrefusal"] = {
                "local_violation": g_ben,
                "linearized_slack": lin_slack_ben,
                "bound": lin_slack_ben + (L_g / 2.0) * consensus_dist_sq,
            }

    return bounds
