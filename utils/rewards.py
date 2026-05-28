"""
Reward utilities for REINFORCE with learned-scalar baseline.

The paper uses:
  R_i = -MSE_final(recon_T_i, target_i)    per-sample (B,)
  baseline = trainable scalar nn.Parameter, trained at 150× actor lr via MSE
  advantage_i = R_i - baseline.detach()
  policy_loss = -mean(sum_t(log_pi_t) * advantage)

Matches original VRRegressReward.lua + nn.Add(1) scalar baseline.
"""
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor


class LearnedBaseline(nn.Module):
    """
    Trainable scalar baseline initialized to 0.

    Trained via MSE against realized per-sample rewards at 150× actor lr,
    matching the original nn.Add(1) scalar in Torch7.
    """
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.zeros(1))

    def forward(self) -> Tensor:
        """Returns the scalar baseline value."""
        return self.value


def compute_reinforce_loss(log_probs: List[Tensor],
                           reward: Tensor,
                           baseline: Tensor,
                           entropies: Optional[List[Tensor]] = None,
                           entropy_coef: float = 0.0) -> Tensor:
    """
    Compute REINFORCE policy gradient loss with optional entropy regularization.

    policy_loss = -E[sum_t log_pi_t · advantage] - entropy_coef · mean_t H(pi_t)

    Entropy bonus prevents the actor logits from collapsing to a flat distribution
    (which would still produce a deterministic argmax). When the reward signal
    across actions is tiny (as it is for tile-resize viewgrids on small datasets),
    the policy collapses without this regularization.

    Args:
        log_probs:    list of (B,) tensors, one per action timestep (T-1 entries)
        reward:       (B,) per-sample reward R_i = -MSE_final_i  (higher = better)
        baseline:     scalar tensor from LearnedBaseline() (detached for advantage)
        entropies:    optional list of (B,) entropies, one per timestep. If provided
                      and entropy_coef > 0, a -coef·mean(H) term is added.
        entropy_coef: scalar α (paper uses no entropy bonus; we add it as a stabilizer)

    Returns:
        policy_loss: scalar tensor
    """
    if not log_probs:
        return torch.tensor(0.0, requires_grad=False)

    stacked = torch.stack(log_probs, dim=0)              # (T-1, B)
    sum_log_probs = stacked.sum(dim=0)                   # (B,)
    advantage = (reward - baseline.detach().squeeze()).detach()  # (B,)
    pg_loss = -(sum_log_probs * advantage).mean()

    if entropies and entropy_coef > 0.0:
        ent = torch.stack(entropies, dim=0).mean()       # scalar mean over T·B
        pg_loss = pg_loss - entropy_coef * ent
    return pg_loss
