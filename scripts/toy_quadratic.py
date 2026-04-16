"""Synthetic quadratic toy experiment (E01).

Validates the core claim: FedAvg produces post-aggregation violations,
while Fed-HyPCA constrained aggregation preserves feasibility.

No GPU needed — runs on CPU in seconds.
Produces Figure 1 data: "two locally feasible points whose average is infeasible."

Usage:
    python scripts/toy_quadratic.py --output results/toy_quadratic.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def quadratic_loss(x, center, scale=1.0):
    """Simple quadratic loss: f(x) = scale * ||x - center||^2"""
    return scale * np.sum((x - center) ** 2)


def constraint_value(x, normal, threshold):
    """Linear constraint: g(x) = normal^T x - threshold <= 0"""
    return np.dot(normal, x) - threshold


def run_toy_experiment(d=10, n_clients=4, n_rounds=30, seed=42):
    """Run the toy quadratic experiment.

    Setup:
    - d-dimensional parameter space
    - n_clients clients, each with a quadratic loss and linear constraints
    - Client optima are spread out; some constraints conflict
    - FedAvg averages → violates strict client constraints
    - Fed-HyPCA constrained aggregation → respects constraints

    Returns:
        dict with per-round metrics for FedAvg and Fed-HyPCA
    """
    rng = np.random.RandomState(seed)

    # Client loss centers (spread out in parameter space)
    centers = rng.randn(n_clients, d) * 2.0

    # Client constraints: each client has 2 linear constraints
    # Designed so that the average of local optima violates some constraints
    constraints = []
    for i in range(n_clients):
        client_constraints = []
        for _ in range(2):
            normal = rng.randn(d)
            normal = normal / np.linalg.norm(normal)
            # Threshold set so local optimum is feasible but average may not be
            threshold = np.dot(normal, centers[i]) + 0.5 + rng.rand() * 0.5
            client_constraints.append({"normal": normal, "threshold": threshold})
        constraints.append(client_constraints)

    # === FedAvg ===
    fedavg_history = []
    theta_fedavg = np.zeros(d)
    local_params_fedavg = [np.zeros(d) for _ in range(n_clients)]

    for r in range(n_rounds):
        # Local training: gradient descent toward local optimum
        for i in range(n_clients):
            u = local_params_fedavg[i].copy()
            for _ in range(5):  # local steps
                grad = 2.0 * (u - centers[i]) + 0.1 * (u - theta_fedavg)
                u -= 0.1 * grad
            local_params_fedavg[i] = u

        # FedAvg aggregation
        theta_fedavg = np.mean(local_params_fedavg, axis=0)

        # Compute violations
        max_viol = 0.0
        total_viol = 0.0
        n_violated = 0
        for i in range(n_clients):
            for c in constraints[i]:
                v = constraint_value(theta_fedavg, c["normal"], c["threshold"])
                if v > 0:
                    n_violated += 1
                    total_viol += v
                    max_viol = max(max_viol, v)

        # Local violations (personalized)
        local_max_viol = 0.0
        for i in range(n_clients):
            for c in constraints[i]:
                v = constraint_value(local_params_fedavg[i], c["normal"], c["threshold"])
                local_max_viol = max(local_max_viol, max(0, v))

        fedavg_history.append({
            "round": r,
            "agg_viol": total_viol / (n_clients * 2),
            "worst_viol": max_viol,
            "n_violated": n_violated,
            "pers_viol": local_max_viol,
            "consensus_dist": np.mean([
                np.sum((theta_fedavg - u) ** 2) for u in local_params_fedavg
            ]),
        })

    # === Fed-HyPCA constrained aggregation ===
    hypca_history = []
    theta_hypca = np.zeros(d)
    local_params_hypca = [np.zeros(d) for _ in range(n_clients)]
    lambdas = [[0.0, 0.0] for _ in range(n_clients)]  # dual variables

    for r in range(n_rounds):
        # Local primal-dual training
        for i in range(n_clients):
            u = local_params_hypca[i].copy()
            for _ in range(5):
                # Primal gradient: loss + proximal + constraint penalties
                grad = 2.0 * (u - centers[i]) + 0.1 * (u - theta_hypca)
                for j, c in enumerate(constraints[i]):
                    g_val = constraint_value(u, c["normal"], c["threshold"])
                    grad += lambdas[i][j] * c["normal"]

                u -= 0.1 * grad

                # Dual update
                for j, c in enumerate(constraints[i]):
                    g_val = constraint_value(u, c["normal"], c["threshold"])
                    lambdas[i][j] = max(0, lambdas[i][j] + 0.05 * g_val)

            local_params_hypca[i] = u

        # Constrained aggregation (projected gradient descent)
        theta_candidate = np.mean(local_params_hypca, axis=0)

        # Project to reduce constraint violations
        for _ in range(10):
            grad = np.zeros(d)
            # Consensus gradient
            for i in range(n_clients):
                grad += (2.0 / n_clients) * (theta_candidate - local_params_hypca[i])

            # Constraint penalty gradient
            for i in range(n_clients):
                for c in constraints[i]:
                    g_val = constraint_value(theta_candidate, c["normal"], c["threshold"])
                    if g_val > 0:
                        grad += 1.0 * c["normal"]  # push away from violation

            theta_candidate -= 0.05 * grad

        theta_hypca = theta_candidate

        # Compute violations
        max_viol = 0.0
        total_viol = 0.0
        n_violated = 0
        for i in range(n_clients):
            for c in constraints[i]:
                v = constraint_value(theta_hypca, c["normal"], c["threshold"])
                if v > 0:
                    n_violated += 1
                    total_viol += v
                    max_viol = max(max_viol, v)

        local_max_viol = 0.0
        for i in range(n_clients):
            for c in constraints[i]:
                v = constraint_value(local_params_hypca[i], c["normal"], c["threshold"])
                local_max_viol = max(local_max_viol, max(0, v))

        # Theorem bound: g(θ) ≤ ξ + L_g/2 ||θ - u_i||²
        bound_values = []
        for i in range(n_clients):
            dist_sq = np.sum((theta_hypca - local_params_hypca[i]) ** 2)
            for c in constraints[i]:
                local_g = constraint_value(local_params_hypca[i], c["normal"], c["threshold"])
                slack = max(0, local_g)
                bound = slack + 0.5 * dist_sq  # L_g = 1.0
                actual = constraint_value(theta_hypca, c["normal"], c["threshold"])
                bound_values.append({
                    "actual": float(actual),
                    "bound": float(bound),
                    "bound_holds": actual <= bound + 1e-6,
                })

        hypca_history.append({
            "round": r,
            "agg_viol": total_viol / (n_clients * 2),
            "worst_viol": max_viol,
            "n_violated": n_violated,
            "pers_viol": local_max_viol,
            "consensus_dist": np.mean([
                np.sum((theta_hypca - u) ** 2) for u in local_params_hypca
            ]),
            "bound_holds_fraction": np.mean([
                b["bound_holds"] for b in bound_values
            ]),
        })

    return {
        "fedavg": fedavg_history,
        "fedhypca": hypca_history,
        "setup": {
            "d": d,
            "n_clients": n_clients,
            "n_rounds": n_rounds,
            "seed": seed,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--n_clients", type=int, default=4)
    parser.add_argument("--n_rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/toy_quadratic.json")
    args = parser.parse_args()

    results = run_toy_experiment(
        d=args.d, n_clients=args.n_clients,
        n_rounds=args.n_rounds, seed=args.seed,
    )

    # Print summary
    fa_final = results["fedavg"][-1]
    hp_final = results["fedhypca"][-1]

    print("=" * 60)
    print("TOY QUADRATIC EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"{'Metric':<25} {'FedAvg':>12} {'Fed-HyPCA':>12}")
    print("-" * 50)
    print(f"{'AggViol':<25} {fa_final['agg_viol']:>12.4f} {hp_final['agg_viol']:>12.4f}")
    print(f"{'WorstViol':<25} {fa_final['worst_viol']:>12.4f} {hp_final['worst_viol']:>12.4f}")
    print(f"{'N Violated':<25} {fa_final['n_violated']:>12d} {hp_final['n_violated']:>12d}")
    print(f"{'PersViol':<25} {fa_final['pers_viol']:>12.4f} {hp_final['pers_viol']:>12.4f}")
    print(f"{'ConsensusDist':<25} {fa_final['consensus_dist']:>12.4f} {hp_final['consensus_dist']:>12.4f}")
    if "bound_holds_fraction" in hp_final:
        print(f"{'Bound Holds %':<25} {'N/A':>12} {hp_final['bound_holds_fraction']:>12.2%}")
    print("=" * 60)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
