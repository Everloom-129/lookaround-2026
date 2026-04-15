"""
Actor: 3-layer policy MLP that maps (aggregate code, relative position, time, abs_elev) → action logits.

Input:
  a_t        (B, 256)  — LSTM aggregate code
  rel_pos    (B, 2)    — (Δelev_norm, Δazim_norm) cumulative from starting position
  time_frac  (B, 1)    — t / T  (normalized step index)
  abs_elev   (B, 1)    — absolute elevation index / (n_elev - 1)
  → total input dim = 260

Matches original: act_ipsz = hiddenSize + 2 + 1 + 1 = 260.

Output:
  logits (B, K)  — raw unnormalized logits over K=14 actions
"""
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self, d_hidden: int = 256, n_actions: int = 14):
        super().__init__()
        d_in = d_hidden + 2 + 1 + 1  # hidden + rel_pos(2) + time_frac(1) + abs_elev(1)
        self.net = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, n_actions),
        )

    def forward(self, a_t: Tensor, rel_pos: Tensor,
                time_frac: Tensor, abs_elev: Tensor) -> Tensor:
        """
        Args:
            a_t:       (B, d_hidden)
            rel_pos:   (B, 2)
            time_frac: (B, 1)
            abs_elev:  (B, 1)
        Returns:
            logits: (B, n_actions)
        """
        if time_frac.dim() == 0:
            time_frac = time_frac.expand(a_t.shape[0], 1)
        elif time_frac.dim() == 1:
            time_frac = time_frac.unsqueeze(1)
        if abs_elev.dim() == 1:
            abs_elev = abs_elev.unsqueeze(1)
        x = torch.cat([a_t, rel_pos, time_frac, abs_elev], dim=1)
        return self.net(x)

    def get_action(self, logits: Tensor,
                   deterministic: bool = False):
        """
        Sample or select action from logits.

        Returns:
            action: (B,) int64 action indices
            log_prob: (B,) log probabilities
            dist: Categorical distribution
        """
        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, dist
