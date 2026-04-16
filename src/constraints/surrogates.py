"""Differentiable constraint surrogates for Fed-HyPCA.

Must-refuse constraint g_{i,k}^ref(u_i):
    g = E_{x~U_{i,k}} [ τ log(1 + exp((α_{i,k} - r(x))/τ)) ] - ε
    Violated when refusal probability drops below floor α.

Over-refusal constraint g_i^ben(u_i):
    g = E_{x~B_i} [ τ log(1 + exp((r(x) - β_i)/τ)) ] - ε_ben
    Violated when refusal probability on benign prompts exceeds cap β.
"""

import torch


def must_refuse_constraint(
    refusal_scores: torch.Tensor,
    alpha: float,
    tau: float = 0.1,
    epsilon: float = 0.05,
) -> torch.Tensor:
    """Compute must-refuse constraint value g_{i,k}^ref.

    Args:
        refusal_scores: (n,) refusal probabilities for unsafe prompts in category k
        alpha: refusal floor (e.g., 0.90 means must refuse ≥90%)
        tau: softplus temperature
        epsilon: slack tolerance

    Returns:
        Scalar constraint value. Negative = satisfied, positive = violated.
    """
    if refusal_scores.numel() == 0:
        return torch.tensor(0.0, device=refusal_scores.device)

    # τ log(1 + exp((α - r(x))/τ)) is a smooth approximation of max(0, α - r(x))
    violations = tau * torch.log1p(torch.exp((alpha - refusal_scores) / tau))
    return violations.mean() - epsilon


def overrefusal_constraint(
    refusal_scores: torch.Tensor,
    beta: float,
    tau: float = 0.1,
    epsilon: float = 0.05,
) -> torch.Tensor:
    """Compute over-refusal constraint value g_i^ben.

    Args:
        refusal_scores: (n,) refusal probabilities for benign prompts
        beta: refusal cap (e.g., 0.10 means max 10% refusal on benign)
        tau: softplus temperature
        epsilon: slack tolerance

    Returns:
        Scalar constraint value. Negative = satisfied, positive = violated.
    """
    if refusal_scores.numel() == 0:
        return torch.tensor(0.0, device=refusal_scores.device)

    violations = tau * torch.log1p(torch.exp((refusal_scores - beta) / tau))
    return violations.mean() - epsilon


def compute_constraint_jacobian(
    constraint_value: torch.Tensor,
    model_params: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Compute Jacobian of constraint w.r.t. model parameters.

    Used for the server-side linearized constrained aggregation.

    Args:
        constraint_value: scalar constraint value (must have grad_fn)
        model_params: list of parameter tensors

    Returns:
        List of gradient tensors (same shapes as model_params)
    """
    grads = torch.autograd.grad(
        constraint_value,
        model_params,
        create_graph=False,
        retain_graph=True,
        allow_unused=True,
    )
    return [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, model_params)]


class DualVariables:
    """Manages dual variables (Lagrange multipliers) for one organization.

    λ_{i,k} for must-refuse constraints (one per active category)
    ν_i for over-refusal constraint
    """

    def __init__(self, must_refuse_indices: list[int], device: torch.device = None):
        self.must_refuse_indices = must_refuse_indices
        self.device = device or torch.device("cpu")

        # Initialize dual variables at 0 (except nu which starts positive to prevent refusal collapse)
        self.lambdas = {
            k: torch.tensor(0.0, device=self.device, requires_grad=False)
            for k in must_refuse_indices
        }
        # Initialize nu > 0 to immediately penalize over-refusal
        self.nu = torch.tensor(0.5, device=self.device, requires_grad=False)

    def update(
        self,
        constraint_values: dict[int, float],
        overrefusal_value: float,
        eta_lambda: float,
        eta_nu: float,
    ):
        """Primal-dual update: λ ← [λ + η_λ g]_+"""
        for k in self.must_refuse_indices:
            if k in constraint_values:
                new_val = self.lambdas[k] + eta_lambda * constraint_values[k]
                self.lambdas[k] = torch.clamp(new_val, min=0.0)

        new_nu = self.nu + eta_nu * overrefusal_value
        self.nu = torch.clamp(new_nu, min=0.0)

    def get_lambda(self, k: int) -> torch.Tensor:
        return self.lambdas.get(k, torch.tensor(0.0, device=self.device))

    def freeze(self):
        """Freeze dual variables (for ablation A12)."""
        for k in self.lambdas:
            self.lambdas[k] = self.lambdas[k].detach()
        self.nu = self.nu.detach()

    def state_dict(self) -> dict:
        return {
            "lambdas": {k: v.item() for k, v in self.lambdas.items()},
            "nu": self.nu.item(),
        }

    def load_state_dict(self, state: dict):
        for k, v in state["lambdas"].items():
            self.lambdas[int(k)] = torch.tensor(v, device=self.device)
        self.nu = torch.tensor(state["nu"], device=self.device)
