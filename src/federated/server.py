"""Federated server orchestrating training rounds.

Manages:
- Global model state (θ)
- Client selection and communication
- Aggregation strategy dispatch
- Round-level logging and checkpointing
"""

import json
import os
import time
from collections import OrderedDict
from pathlib import Path

import torch
from tqdm import tqdm

from src.federated.aggregation import (
    fedavg_aggregate,
    qffl_aggregate,
    scaffold_aggregate,
    fedhypca_constrained_aggregate,
    compute_aggregation_violation_bound,
)
from src.federated.client import FedClient


def _any_nonzero_jacobians(constraint_jacobians, overrefusal_jacobians):
    """Check if any Jacobian tensor is non-zero."""
    for jacs in constraint_jacobians:
        for k, j in jacs.items():
            if isinstance(j, torch.Tensor) and j.norm().item() > 1e-10:
                return True
    for j in overrefusal_jacobians:
        if isinstance(j, torch.Tensor) and j.norm().item() > 1e-10:
            return True
    return False


class FedServer:
    """Federated learning server for Fed-HyPCA."""

    def __init__(self, config, model, clients: list[FedClient]):
        self.config = config
        self.model = model
        self.clients = clients
        self.global_state = model.get_lora_state_dict()
        self.round_idx = 0

        # SCAFFOLD control variates
        if config.training.aggregation == "scaffold":
            self.server_control = OrderedDict(
                {k: torch.zeros_like(v) for k, v in self.global_state.items()}
            )
        else:
            self.server_control = None

        # Logging
        self.history = []

    def run_round(self) -> dict:
        """Execute one federated round.

        Returns:
            Round metrics dict
        """
        t_start = time.time()
        tc = self.config.training
        agg_type = tc.aggregation

        # === Client local training ===
        client_states = []
        client_losses = []
        client_constraint_values = []
        client_constraint_jacobians = []
        client_overrefusal_values = []
        client_overrefusal_jacobians = []

        for client in tqdm(self.clients, desc="  Clients", leave=False, unit="client"):
            # Each client trains locally
            local_state = client.local_train(
                global_state=self.global_state,
                round_idx=self.round_idx,
                aggregation_type=agg_type,
            )
            client_states.append(local_state)
            client_losses.append(client.get_loss())

            # Collect constraint info for constrained aggregation
            client_constraint_values.append(client.last_constraint_values)
            client_constraint_jacobians.append(client.last_constraint_jacobians)
            client_overrefusal_values.append(client.last_overrefusal_value)
            client_overrefusal_jacobians.append(client.last_overrefusal_jacobian)

        # === Server aggregation ===
        # Ablation A1: use_constrained_aggregation controls server behavior
        use_constrained = self.config.use_constrained_aggregation and agg_type == "fedhypca"

        if agg_type in ("fedavg", "fedprox", "ditto", "pfedme",
                        "fedavg_dual", "ditto_dual"):
            self.global_state = fedavg_aggregate(client_states)

        elif agg_type == "qffl":
            self.global_state = qffl_aggregate(
                client_states, client_losses, q=tc.qffl_q
            )

        elif agg_type == "scaffold":
            client_controls = [
                c.control_variate or OrderedDict(
                    {k: torch.zeros_like(v) for k, v in self.global_state.items()}
                )
                for c in self.clients
            ]
            self.global_state, self.server_control = scaffold_aggregate(
                client_states, client_controls, self.server_control
            )

        elif agg_type == "fedhypca":
            if use_constrained:
                # Auto-detect zero Jacobians: if all are zero, force scalar path
                has_jacobians = _any_nonzero_jacobians(
                    client_constraint_jacobians, client_overrefusal_jacobians
                )
                actual_use_jacobian = self.config.use_jacobian_correction and has_jacobians
                if self.config.use_jacobian_correction and not has_jacobians:
                    print(f"  [Round {self.round_idx}] Jacobians all zero, using scalar reweighting")

                # Full constrained aggregation
                self.global_state = fedhypca_constrained_aggregate(
                    client_states=client_states,
                    client_constraint_values=client_constraint_values,
                    client_constraint_jacobians=client_constraint_jacobians,
                    client_overrefusal_values=client_overrefusal_values,
                    client_overrefusal_jacobians=client_overrefusal_jacobians,
                    beta=tc.beta,
                    beta_b=tc.beta_b,
                    use_jacobian=actual_use_jacobian,
                    use_slack=self.config.use_slack_variables,
                    scalar_beta=tc.scalar_reweight_beta,
                )
            else:
                # Ablation A1: Fall back to FedAvg server (but keep local dual training)
                self.global_state = fedavg_aggregate(client_states)

        else:
            raise ValueError(f"Unknown aggregation type: {agg_type}")

        # === Compute round metrics ===
        round_metrics = {
            "round": self.round_idx,
            "time_seconds": time.time() - t_start,
            "avg_client_loss": sum(client_losses) / len(client_losses),
            "client_losses": {
                c.org_id: loss for c, loss in zip(self.clients, client_losses)
            },
        }

        # Constraint satisfaction summary
        for i, client in enumerate(self.clients):
            org_id = client.org_id
            round_metrics[f"{org_id}_constraint_values"] = client_constraint_values[i]
            round_metrics[f"{org_id}_overrefusal"] = client_overrefusal_values[i]
            round_metrics[f"{org_id}_duals"] = client.duals.state_dict()

        # Compute violation bound if using constrained aggregation
        if agg_type == "fedhypca" and self.config.use_constrained_aggregation:
            bounds = compute_aggregation_violation_bound(
                self.global_state,
                client_states,
                client_constraint_values,
                client_constraint_jacobians,
                client_overrefusal_values,
                client_overrefusal_jacobians,
            )
            round_metrics["violation_bounds"] = {
                str(k): v for k, v in bounds.items()
            }

        # Consensus distance
        global_flat = torch.cat([
            v.float().reshape(-1) for v in self.global_state.values()
        ])
        consensus_dists = []
        for cs in client_states:
            cs_flat = torch.cat([v.float().reshape(-1) for v in cs.values()])
            consensus_dists.append(
                torch.sum((global_flat - cs_flat) ** 2).item()
            )
        round_metrics["consensus_dist_avg"] = sum(consensus_dists) / len(consensus_dists)
        round_metrics["consensus_dist_max"] = max(consensus_dists)

        self.history.append(round_metrics)
        self.round_idx += 1

        return round_metrics

    def train(self, num_rounds: int = None) -> list[dict]:
        """Run full federated training.

        Args:
            num_rounds: override config num_rounds

        Returns:
            List of per-round metrics
        """
        if num_rounds is None:
            num_rounds = self.config.training.num_rounds

        print(f"Starting federated training: {num_rounds} rounds, "
              f"{len(self.clients)} clients, aggregation={self.config.training.aggregation}")

        for r in range(num_rounds):
            metrics = self.run_round()

            # Print progress
            if r % 5 == 0 or r == num_rounds - 1:
                avg_loss = metrics["avg_client_loss"]
                cons_dist = metrics["consensus_dist_avg"]
                print(f"  Round {r}/{num_rounds}: "
                      f"avg_loss={avg_loss:.4f}, "
                      f"consensus_dist={cons_dist:.6f}")

                # Print constraint satisfaction
                for client in self.clients:
                    org_id = client.org_id
                    cv = metrics.get(f"{org_id}_constraint_values", {})
                    violated = sum(1 for v in cv.values() if v > 0)
                    total = len(cv)
                    print(f"    {org_id}: {total - violated}/{total} constraints satisfied")

        return self.history

    def save_checkpoint(self, path: str):
        """Save server state and training history."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "global_state": self.global_state,
            "round_idx": self.round_idx,
            "config": {
                "aggregation": self.config.training.aggregation,
                "num_rounds": self.config.training.num_rounds,
            },
            "client_duals": {
                c.org_id: c.duals.state_dict() for c in self.clients
            },
            "client_states": {
                c.org_id: c.last_local_state for c in self.clients if c.last_local_state is not None
            },
        }
        torch.save(checkpoint, path)

        # Save history as JSON
        history_path = path.replace(".pt", "_history.json")
        # Convert non-serializable values
        serializable_history = []
        for h in self.history:
            sh = {}
            for k, v in h.items():
                if isinstance(v, (int, float, str, bool)):
                    sh[k] = v
                elif isinstance(v, dict):
                    sh[k] = {
                        str(kk): (vv if isinstance(vv, (int, float, str, bool, dict, list)) else str(vv))
                        for kk, vv in v.items()
                    }
                else:
                    sh[k] = str(v)
            serializable_history.append(sh)

        with open(history_path, "w") as f:
            json.dump(serializable_history, f, indent=2)

    def load_checkpoint(self, path: str):
        """Load server state from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        self.global_state = checkpoint["global_state"]
        self.round_idx = checkpoint["round_idx"]

        if "client_duals" in checkpoint:
            for client in self.clients:
                if client.org_id in checkpoint["client_duals"]:
                    client.duals.load_state_dict(
                        checkpoint["client_duals"][client.org_id]
                    )
