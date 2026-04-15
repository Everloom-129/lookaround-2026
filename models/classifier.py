"""
Categorization head for active classification (Fig 5 policy transfer).

A linear classifier on top of the LSTM hidden state h_T, trained
with cross-entropy loss using scene category labels.
"""
import torch
import torch.nn as nn
from torch import Tensor


class CategorizationHead(nn.Module):
    """
    Linear classifier: d_hidden → n_classes.

    Applied to the final LSTM hidden state h_T after T steps of exploration.
    Trained on top of frozen completion-policy features for policy transfer.
    """

    def __init__(self, d_hidden: int = 256, n_classes: int = 26):
        super().__init__()
        self.fc = nn.Linear(d_hidden, n_classes)

    def forward(self, h_T: Tensor) -> Tensor:
        """
        Args:
            h_T: (B, d_hidden) — final LSTM hidden state
        Returns:
            logits: (B, n_classes)
        """
        return self.fc(h_T)
