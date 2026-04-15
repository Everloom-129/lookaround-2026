"""
LocationSensor: small MLP that encodes proprioceptive metadata into a 16D vector.

Input p_t = [rel_elev_norm, rel_azim_norm, t/T, abs_elev_norm]  — 4 inputs
  - rel_elev_norm  = cumulative elevation offset from start / (n_elev - 1)
  - rel_azim_norm  = cumulative azimuth offset from start / n_azim
  - t/T            = normalized step index (0 at first step)
  - abs_elev_norm  = elev_idx / (n_elev - 1)

Matches original lookaround.lua: location_ipsz = 2 + 1 + 1 (rel_pos + time + knownElev).
At t=0: rel_elev = rel_azim = 0, t/T = 0.
"""
import torch.nn as nn
from torch import Tensor


class LocationSensor(nn.Module):
    def __init__(self, d_in: int = 4, d_out: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, p: Tensor) -> Tensor:
        """
        Args:
            p: (B, d_in)
        Returns:
            (B, d_out)
        """
        return self.net(p)
