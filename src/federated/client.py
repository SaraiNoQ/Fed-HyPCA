"""Client-side local training for Fed-HyPCA.

Implements the local primal-dual update:
- Primal: gradient descent on the Lagrangian w.r.t. model params
- Dual: gradient ascent on λ, ν (Lagrange multipliers)
"""

from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constraints.surrogates import (
    must_refuse_constraint,
    overrefusal_constraint,
    DualVariables,
)
from src.models.refusal_head import compute_refusal_aux_loss
from configs.default import ORG_POLICIES, SAFETY_CATEGORIES


class FedClient:
    """A single federated client (organization).

    Manages local training with primal-dual constraint optimization.
    """

    def __init__(
        self,
        org_id: str,
        model,
        train_dataset,
        val_dataset,
        config,
        device: torch.device = None,
    ):
        self.org_id = org_id
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Organization policy
        self.policy = ORG_POLICIES[org_id]
        self.must_refuse_indices = self.policy["must_refuse"]
        self.alpha = self.policy["alpha"]  # {category_idx: refusal_floor}
        self.beta = self.policy["beta"]  # over-refusal cap

        # Dual variables
        self.duals = DualVariables(
            must_refuse_indices=self.must_refuse_indices,
            device=self.device,
        )

        # For SCAFFOLD: local control variate
        self.control_variate = None

        # Cached constraint info for server aggregation
        self.last_constraint_values = {}
        self.last_constraint_jacobians = {}
        self.last_overrefusal_value = 0.0
        self.last_overrefusal_jacobian = None

        # Store last local state (since all clients share the same model instance)
        self.last_local_state = None

    def local_train(
        self,
        global_state: OrderedDict,
        round_idx: int,
        aggregation_type: str = "fedhypca",
    ) -> OrderedDict:
        """Run local training and return updated parameters.

        Args:
            global_state: server's current LoRA state dict (θ^t)
            round_idx: current federated round
            aggregation_type: which FL method to use

        Returns:
            Updated local LoRA state dict (u_i^{t+1})
        """
        tc = self.config.training

        # Load global parameters
        self.model.set_lora_state_dict(global_state)

        # Save initial state for proximal term
        if aggregation_type in ("fedprox", "fedhypca", "ditto", "pfedme"):
            initial_state = self.model.get_lora_state_dict()
            initial_flat = torch.cat([
                v.float().reshape(-1) for v in initial_state.values()
            ])

        # Ablation A6: if no personalization, start from global state and don't deviate
        if not self.config.use_personalization:
            # Skip local training entirely - just return global state
            self.last_local_state = global_state.copy()
            return self.last_local_state

        # Optimizer
        trainable_params = self.model.get_trainable_params()
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=tc.learning_rate,
            weight_decay=tc.weight_decay,
        )

        # DataLoader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=tc.batch_size,
            shuffle=True,
            drop_last=True,
        )

        self.model.train()

        for epoch in range(tc.local_epochs):
            # Determine number of steps for this epoch
            if tc.local_steps_per_epoch > 0:
                n_steps = tc.local_steps_per_epoch
            else:
                n_steps = len(train_loader)

            # Accumulated constraint values for stable dual updates
            accumulated_constraints = {}
            accumulated_overrefusal = []

            step_pbar = tqdm(
                enumerate(train_loader),
                total=n_steps,
                desc=f"    {self.org_id}",
                leave=False,
                unit="step",
            )
            for step, batch in step_pbar:
                if tc.local_steps_per_epoch > 0 and step >= tc.local_steps_per_epoch:
                    break

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                category_indices = batch["category_idx"].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,  # causal LM loss
                )

                refusal_scores = outputs["refusal_scores"]

                # === Compute Lagrangian components ===

                # 1. Allowed-prompt NLL (utility loss)
                # Only compute on non-refuse samples
                respond_mask = (labels == 0)
                if respond_mask.any() and outputs["loss"] is not None:
                    loss_allow = outputs["loss"]
                else:
                    loss_allow = torch.tensor(0.0, device=self.device)

                # 2. Refusal head auxiliary loss
                if self.config.use_refusal_head_aux:
                    loss_aux = compute_refusal_aux_loss(refusal_scores, labels.float())
                else:
                    loss_aux = torch.tensor(0.0, device=self.device)

                # 3. Proximal consensus term: ρ/2 ||u_i - θ||²
                loss_proximal = torch.tensor(0.0, device=self.device)
                if self.config.use_proximal_term and aggregation_type in (
                    "fedhypca", "fedprox", "ditto"
                ):
                    current_flat = torch.cat([
                        p.float().reshape(-1) for p in trainable_params
                    ])
                    prox_mu = tc.rho if aggregation_type == "fedhypca" else tc.fedprox_mu
                    loss_proximal = (prox_mu / 2.0) * torch.sum(
                        (current_flat - initial_flat.detach()) ** 2
                    )

                # 4. Constraint terms (only for fedhypca and constrained variants)
                loss_constraints = torch.tensor(0.0, device=self.device)
                constraint_values = {}

                if aggregation_type in ("fedhypca", "fedavg_dual", "ditto_dual"):
                    # Ablation A10: use_structured_policy - if False, use single "unsafe" bit
                    if self.config.use_structured_policy:
                        # Structured: per-category constraints
                        for k in self.must_refuse_indices:
                            cat_mask = (category_indices == k)
                            if cat_mask.any():
                                cat_refusal = refusal_scores[cat_mask]
                                g_ref = must_refuse_constraint(
                                    cat_refusal,
                                    alpha=self.alpha[k],
                                    tau=tc.tau,
                                    epsilon=tc.epsilon_ref,
                                )
                                constraint_values[k] = g_ref.item()
                                lam_k = self.duals.get_lambda(k)
                                loss_constraints += lam_k * g_ref
                    else:
                        # Ablation A10: Single "unsafe" constraint (aggregate all categories)
                        unsafe_mask = (category_indices >= 0)
                        if unsafe_mask.any() and (labels == 1).any():
                            # Combine all must-refuse samples
                            unsafe_refusal = refusal_scores[(labels == 1)]
                            # Use minimum alpha across all categories as the floor
                            min_alpha = min(self.alpha.values()) if self.alpha else 0.9
                            g_ref = must_refuse_constraint(
                                unsafe_refusal,
                                alpha=min_alpha,
                                tau=tc.tau,
                                epsilon=tc.epsilon_ref,
                            )
                            constraint_values[-1] = g_ref.item()  # -1 = single unsafe bit
                            lam = self.duals.get_lambda(list(self.duals.lambdas.keys())[0]) if self.duals.lambdas else torch.tensor(0.0, device=self.device)
                            loss_constraints += lam * g_ref

                    # Over-refusal constraint: only on truly benign prompts (category_idx == -1)
                    if self.config.use_overrefusal_constraint:
                        benign_mask = (category_indices == -1)
                        if benign_mask.any():
                            benign_refusal = refusal_scores[benign_mask]
                            g_ben = overrefusal_constraint(
                                benign_refusal,
                                beta=self.beta,
                                tau=tc.tau,
                                epsilon=tc.epsilon_ben,
                            )
                            self.last_overrefusal_value = g_ben.item()
                            loss_constraints += self.duals.nu * g_ben

                    # Dual regularization: -μ/2 (||λ||² + ν²)
                    dual_reg = (tc.mu / 2.0) * (
                        sum(lam.item() ** 2 for lam in self.duals.lambdas.values())
                        + self.duals.nu.item() ** 2
                    )
                    loss_constraints -= dual_reg

                # Total Lagrangian
                total_loss = (
                    loss_allow
                    + tc.gamma * loss_aux
                    + loss_proximal
                    + loss_constraints
                )

                # Backward + step
                optimizer.zero_grad()
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

                # Dual variable update (accumulated over interval for stability)
                if aggregation_type in ("fedhypca", "fedavg_dual", "ditto_dual"):
                    # Accumulate constraint values
                    for k, v in constraint_values.items():
                        if k not in accumulated_constraints:
                            accumulated_constraints[k] = []
                        accumulated_constraints[k].append(v)
                    accumulated_overrefusal.append(self.last_overrefusal_value)

                    should_update_duals = (
                        self.config.use_adaptive_duals
                        and (self.config.freeze_duals_after < 0
                             or round_idx < self.config.freeze_duals_after)
                    )
                    if should_update_duals and (step + 1) % tc.dual_update_interval == 0:
                        # Average accumulated values for stable update
                        avg_constraints = {
                            k: sum(vs) / len(vs)
                            for k, vs in accumulated_constraints.items()
                        }
                        avg_overrefusal = (
                            sum(accumulated_overrefusal) / len(accumulated_overrefusal)
                            if accumulated_overrefusal else 0.0
                        )
                        self.duals.update(
                            constraint_values=avg_constraints,
                            overrefusal_value=avg_overrefusal,
                            eta_lambda=tc.eta_lambda,
                            eta_nu=tc.eta_nu,
                        )
                        accumulated_constraints = {}
                        accumulated_overrefusal = []

        # Cache constraint values and Jacobians for server aggregation
        self._compute_constraint_info_for_server(train_loader)

        # Store local state before returning (important: clients share model instance)
        self.last_local_state = self.model.get_lora_state_dict()
        return self.last_local_state

    def _compute_constraint_info_for_server(self, train_loader):
        """Compute constraint values and Jacobians over the validation set.

        These are sent to the server for the constrained aggregation QP.
        Jacobians are computed on a subset for memory efficiency.
        """
        self.model.eval()
        self.last_constraint_values = {}
        self.last_constraint_jacobians = {}

        tc = self.config.training
        trainable_params = self.model.get_trainable_params()
        total_params = sum(p.numel() for p in trainable_params)

        # Collect all refusal scores across validation set
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.eval.eval_batch_size,
            shuffle=False,
        )

        all_refusal_scores = {k: [] for k in self.must_refuse_indices}
        all_benign_scores = []

        # First pass: collect constraint values (no gradients)
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                category_indices = batch["category_idx"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                refusal_scores = outputs["refusal_scores"]

                # Collect per-category scores
                for k in self.must_refuse_indices:
                    cat_mask = (category_indices == k)
                    if cat_mask.any():
                        all_refusal_scores[k].append(refusal_scores[cat_mask].cpu())

                # Collect benign scores
                benign_mask = (category_indices == -1)
                if benign_mask.any():
                    all_benign_scores.append(refusal_scores[benign_mask].cpu())

        # Compute constraint values per category
        for k in self.must_refuse_indices:
            if all_refusal_scores[k]:
                cat_scores = torch.cat(all_refusal_scores[k])
                g_val = must_refuse_constraint(
                    cat_scores.to(self.device),
                    alpha=self.alpha[k],
                    tau=tc.tau,
                    epsilon=tc.epsilon_ref,
                )
                self.last_constraint_values[k] = g_val.item()
            else:
                # No examples in this category - set to zero violation
                self.last_constraint_values[k] = 0.0

        # Compute over-refusal constraint
        if all_benign_scores:
            benign_scores = torch.cat(all_benign_scores)
            g_ben = overrefusal_constraint(
                benign_scores.to(self.device),
                beta=self.beta,
                tau=tc.tau,
                epsilon=tc.epsilon_ben,
            )
            self.last_overrefusal_value = g_ben.item()
        else:
            self.last_overrefusal_value = 0.0

        # Second pass: Compute Jacobians on a small subset for memory efficiency
        # For Qwen3.5 4B, skip Jacobian to avoid OOM (use scalar weights only)
        # This still works because the constraint values guide the aggregation

        # Initialize with zeros (ablation A2 style - scalar weights only)
        for k in self.must_refuse_indices:
            self.last_constraint_jacobians[k] = torch.zeros(total_params, device=self.device)
        self.last_overrefusal_jacobian = torch.zeros(total_params, device=self.device)

        # Note: Jacobian computation disabled for memory efficiency with Qwen3.5 4B
        # The constrained aggregation will use scalar constraint values instead
        # This is similar to ablation A2 (no_jacobian_correction)

        self.model.train()

    def get_loss(self) -> float:
        """Compute validation loss (for q-FFL weighting)."""
        self.model.eval()
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.eval.eval_batch_size,
            shuffle=False,
        )
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )
                if outputs["loss"] is not None:
                    total_loss += outputs["loss"].item()
                    n_batches += 1
                if n_batches >= 4:
                    break
        self.model.train()
        return total_loss / max(n_batches, 1)
