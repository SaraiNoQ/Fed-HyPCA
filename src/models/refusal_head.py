"""Lightweight refusal head: linear probe on last-token hidden state.

Predicts r(x) ∈ (0,1) = probability that the model should refuse prompt x.
Trained jointly with the main LM via auxiliary BCE loss.
"""

import torch
import torch.nn as nn


class RefusalHead(nn.Module):
    """Linear probe mapping last-token hidden state → refusal probability.

    Architecture: hidden_dim → 1 → sigmoid
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Compute refusal score for each sample in the batch.

        Args:
            hidden_states: (batch, seq_len, hidden_dim) from the base model
            attention_mask: (batch, seq_len) to find last real token

        Returns:
            refusal_scores: (batch,) in (0, 1)
        """
        if attention_mask is not None:
            # Get last non-padding token position
            seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            last_hidden = hidden_states[batch_indices, seq_lengths]  # (batch, hidden_dim)
        else:
            last_hidden = hidden_states[:, -1, :]  # (batch, hidden_dim)

        # Cast to float32 for the linear layer to handle bfloat16 inputs
        last_hidden = last_hidden.float()
        logits = self.linear(last_hidden).squeeze(-1)  # (batch,)
        return torch.sigmoid(logits)

    def get_logits(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Return raw logits (before sigmoid) for gradient computation."""
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            last_hidden = hidden_states[batch_indices, seq_lengths]
        else:
            last_hidden = hidden_states[:, -1, :]

        # Cast to float32 for the linear layer
        last_hidden = last_hidden.float()
        return self.linear(last_hidden).squeeze(-1)  # (batch,)


def compute_refusal_aux_loss(
    refusal_scores: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """BCE auxiliary loss for training the refusal head.

    Args:
        refusal_scores: (batch,) predicted refusal probabilities
        labels: (batch,) binary labels (1=should refuse, 0=should respond)

    Returns:
        Scalar BCE loss
    """
    return nn.functional.binary_cross_entropy(
        refusal_scores,
        labels.float(),
        reduction="mean",
    )
