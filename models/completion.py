"""
CompletionHead: transposed-convolution decoder that maps the LSTM aggregate code
to a full predicted viewgrid of shape (B, N_views, C, H, W).

Architecture (from code-2017/SUN360/SUN360ActiveMod.lua, reconstructor):
  Linear(256 → 1024) → LeakyReLU(0.2)
  Reshape(B, 64, 4, 4)
  ConvTranspose2d(64→256,  5×5, stride=2, pad=2, output_pad=1)  →  8×8  → ReLU
  ConvTranspose2d(256→128, 5×5, stride=2, pad=2, output_pad=1)  → 16×16 → ReLU
  ConvTranspose2d(128→N_views*C, 5×5, stride=2, pad=2, output_pad=1)  → 32×32 → LeakyReLU(0.8)
  Reshape(B, N_views, C, 32, 32)

Note: final activation is LeakyReLU(0.8, inplace=True), not Sigmoid, because targets are
mean-subtracted and can be negative. Sigmoid cannot produce negative outputs.
"""
import torch.nn as nn
from torch import Tensor


class CompletionHead(nn.Module):
    def __init__(self, d_hidden: int = 256, n_views: int = 40,
                 n_channels: int = 3, view_size: int = 32):
        super().__init__()
        self.n_views = n_views
        self.n_channels = n_channels
        self.view_size = view_size

        self.fc = nn.Sequential(
            nn.Linear(d_hidden, 1024),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.deconv = nn.Sequential(
            # 4×4 → 8×8
            nn.ConvTranspose2d(64, 256, kernel_size=5, stride=2,
                               padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            # 8×8 → 16×16
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2,
                               padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            # 16×16 → 32×32
            nn.ConvTranspose2d(128, n_views * n_channels, kernel_size=5, stride=2,
                               padding=2, output_padding=1),
            nn.LeakyReLU(0.8, inplace=True),
        )

    def forward(self, a: Tensor) -> Tensor:
        """
        Args:
            a: (B, d_hidden) — LSTM aggregate code
        Returns:
            (B, N_views, C, view_size, view_size)
        """
        B = a.shape[0]
        x = self.fc(a)                  # (B, 1024)
        x = x.view(B, 64, 4, 4)        # reshape to spatial
        x = self.deconv(x)              # (B, N_views*C, 32, 32)
        x = x.view(B, self.n_views, self.n_channels,
                   self.view_size, self.view_size)
        return x
