"""
CombineModule: fuses ViewEncoder output and LocationSensor output.

  Concat([patch_feat(256), loc_feat(16)]) → (B, 272)
  → Linear(272 → 256) → ReLU
  → Linear(256 → 256) → BatchNorm(256) → Dropout
"""
import torch
import torch.nn as nn
from torch import Tensor


class CombineModule(nn.Module):
    def __init__(self, d_patch: int = 256, d_loc: int = 16,
                 d_out: int = 256, dropout: float = 0.3):
        super().__init__()
        d_in = d_patch + d_loc
        self.net = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, d_out),
            nn.BatchNorm1d(d_out),
            nn.Dropout(p=dropout),
        )

    def forward(self, patch_feat: Tensor, loc_feat: Tensor) -> Tensor:
        """
        Args:
            patch_feat: (B, d_patch)
            loc_feat:   (B, d_loc)
        Returns:
            (B, d_out)
        """
        x = torch.cat([patch_feat, loc_feat], dim=1)
        return self.net(x)
