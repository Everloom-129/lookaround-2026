"""
AgentMemory: single-layer LSTM that aggregates per-step fused features.

  input_size  = 256 (CombineModule output)
  hidden_size = 256
  batch_first = True
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class AgentMemory(nn.Module):
    def __init__(self, d_in: int = 256, d_hidden: int = 256):
        super().__init__()
        self.d_hidden = d_hidden
        self.lstm = nn.LSTM(
            input_size=d_in,
            hidden_size=d_hidden,
            num_layers=1,
            batch_first=True,
        )

    def init_hidden(self, batch_size: int,
                    device: torch.device) -> Tuple[Tensor, Tensor]:
        """Return zero initial (h, c) state."""
        h = torch.zeros(1, batch_size, self.d_hidden, device=device)
        c = torch.zeros(1, batch_size, self.d_hidden, device=device)
        return h, c

    def forward(self, f_t: Tensor,
                state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        One LSTM step.

        Args:
            f_t:   (B, d_in)  — combined sense features at timestep t
            state: ((1, B, d_hidden), (1, B, d_hidden))

        Returns:
            a_t:      (B, d_hidden)  — aggregate code
            new_state: updated (h, c)
        """
        # LSTM expects (B, seq_len, input_size); we step one at a time
        out, new_state = self.lstm(f_t.unsqueeze(1), state)  # out: (B, 1, d_hidden)
        a_t = out.squeeze(1)                                  # (B, d_hidden)
        return a_t, new_state
