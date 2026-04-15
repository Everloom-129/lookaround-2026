"""
Baseline policies for comparison against the learned actor.

RandomPolicy    — uniformly random action from the 14-action set
LargeActionPolicy — always selects the action with the largest magnitude
                    (perimeter of the 3×5 neighborhood grid)
"""
import random
from typing import List, Tuple

import torch
from torch import Tensor


class RandomPolicy:
    """Uniformly random action selection."""

    def __init__(self, n_actions: int = 14):
        self.n_actions = n_actions

    def get_action(self, batch_size: int,
                   device: torch.device) -> Tensor:
        """Returns (B,) int64 random action indices."""
        return torch.randint(0, self.n_actions, (batch_size,), device=device)


class LargeActionPolicy:
    """
    Always selects an action on the perimeter of the 3×5 action grid —
    i.e., actions with the largest displacement magnitude.
    Cycles through all perimeter actions.

    Perimeter of 3×5 (rows={-1,0,1}, cols={-2,-1,0,1,2}) minus center:
    All actions with |d_elev|=1 OR |d_azim|=2, i.e., the outer ring.
    """

    def __init__(self, action_deltas: List[Tuple[int, int]]):
        # Identify perimeter actions (max elev or max azim displacement)
        self._perimeter_indices = [
            i for i, (de, da) in enumerate(action_deltas)
            if abs(de) == 1 or abs(da) == 2
        ]
        self._cycle_idx = 0

    def get_action(self, batch_size: int,
                   device: torch.device) -> Tensor:
        """Returns (B,) int64 action indices, cycling through perimeter actions."""
        action_idx = self._perimeter_indices[
            self._cycle_idx % len(self._perimeter_indices)
        ]
        self._cycle_idx += 1
        return torch.full((batch_size,), action_idx,
                          dtype=torch.long, device=device)
