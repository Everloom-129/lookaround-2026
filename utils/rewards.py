"""
Reward utilities for REINFORCE with EMA baseline.

The paper uses:
  R = -MSE_final(recon_T, target)         (negative final-step reconstruction error)
  baseline = EMA of past R values
  advantage = R - baseline
  policy_loss = -sum(log_probs) * advantage
"""
from typing import List

import torch
from torch import Tensor


def update_ema_baseline(baseline: float, reward: float,
                        alpha: float = 0.9) -> float:
    """
    Update exponential moving average baseline.

    baseline_{new} = alpha * baseline_{old} + (1 - alpha) * reward

    Args:
        baseline: current EMA baseline scalar
        reward:   current episode batch-mean reward
        alpha:    decay factor (0.9 = slow update, 0.1 = fast update)

    Returns:
        Updated baseline scalar.
    """
    return alpha * baseline + (1.0 - alpha) * reward


def compute_reinforce_loss(log_probs: List[Tensor],
                           reward: Tensor,
                           baseline: float) -> Tensor:
    """
    Compute REINFORCE policy gradient loss.

    Uses a scalar EMA baseline for variance reduction.
    Applies the same reward signal to all timesteps' log probs
    (non-myopic: reward is the FINAL step MSE, not per-step).

    Args:
        log_probs: list of (B,) tensors, one per action timestep (T-1 entries)
        reward:    (B,) or scalar tensor — R = -MSE_final (higher = better)
        baseline:  scalar float — EMA baseline

    Returns:
        policy_loss: scalar tensor (to be added to reconstruction loss)
    """
    if not log_probs:
        return torch.tensor(0.0, requires_grad=False)

    # Stack log probs: (T-1, B)
    stacked = torch.stack(log_probs, dim=0)  # (T-1, B)

    # Sum over timesteps for each batch element: (B,)
    sum_log_probs = stacked.sum(dim=0)       # (B,)

    # Advantage (detached — no gradient into reward computation)
    advantage = (reward - baseline).detach()  # (B,) or scalar

    # Policy gradient loss (negative because we want to maximize reward)
    policy_loss = -(sum_log_probs * advantage).mean()
    return policy_loss
