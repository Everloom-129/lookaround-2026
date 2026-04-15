"""
Reward utilities for REINFORCE with learned-scalar baseline.

The paper uses:
  R_i = -MSE_final(recon_T_i, target_i)    per-sample (B,)
  baseline = trainable scalar nn.Parameter, trained at 150× actor lr via MSE
  advantage_i = R_i - baseline.detach()
  policy_loss = -mean(sum_t(log_pi_t) * advantage)

Matches original VRRegressReward.lua + nn.Add(1) scalar baseline.
"""
from typing import List

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
                           baseline: Tensor) -> Tensor:
    """
    Compute REINFORCE policy gradient loss.

    Uses per-sample reward and a learned scalar baseline for variance reduction.
    Applies the same reward signal to all timesteps' log probs (non-myopic).

    Args:
        log_probs: list of (B,) tensors, one per action timestep (T-1 entries)
        reward:    (B,) per-sample reward R_i = -MSE_final_i  (higher = better)
        baseline:  scalar tensor from LearnedBaseline() (already detached for advantage)

    Returns:
        policy_loss: scalar tensor (to be added to actor optimizer step)
    """
    if not log_probs:
        return torch.tensor(0.0, requires_grad=False)

    # Stack log probs: (T-1, B)
    stacked = torch.stack(log_probs, dim=0)  # (T-1, B)

    # Sum over timesteps for each batch element: (B,)
    sum_log_probs = stacked.sum(dim=0)       # (B,)

    # Advantage: detach baseline so no gradient flows into it here
    advantage = (reward - baseline.detach().squeeze()).detach()  # (B,)

    # Policy gradient loss (negative because we maximize reward)
    policy_loss = -(sum_log_probs * advantage).mean()
    return policy_loss
