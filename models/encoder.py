"""
ViewEncoder: 3-layer CNN that encodes a 3×32×32 view into a 256D feature vector.

Architecture (from code-2017/SUN360/SUN360ActiveMod.lua, patchSensor):
  Conv(3→32, 5×5, pad=2) → MaxPool(3×3, stride=2) → ReLU
  Conv(32→32, 5×5, pad=2) → AvgPool(3×3, stride=2) → ReLU
  Conv(32→64, 5×5, pad=2) → AvgPool(3×3, stride=2) → Flatten
  Linear(? → 256)

Spatial sizes for 32×32 input:
  After pool1 (MaxPool 3×3/2): floor((32-3)/2)+1 = 15 → 15×15
  After pool2 (AvgPool 3×3/2): floor((15-3)/2)+1 = 7  → 7×7
  After pool3 (AvgPool 3×3/2): floor((7-3)/2)+1  = 3  → 3×3
  Flattened: 64 × 3 × 3 = 576
"""
import torch
import torch.nn as nn
from torch import Tensor


class ViewEncoder(nn.Module):
    def __init__(self, d_enc: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            # Block 2
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            # Block 3
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
        )
        # Compute flattened dim dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            flat_dim = self.features(dummy).shape[1]

        self.fc = nn.Linear(flat_dim, d_enc)
        self.d_enc = d_enc

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, 3, 32, 32)
        Returns:
            (B, d_enc)
        """
        return self.fc(self.features(x))
