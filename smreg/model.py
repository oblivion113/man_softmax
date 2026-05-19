import torch
import torch.nn as nn

from .config import INPUT_DIM, NUM_CLASSES


class SoftmaxRegression(nn.Module):
    """Single linear layer classifier trained with cross-entropy loss."""

    def __init__(self, d: int = INPUT_DIM, q: int = NUM_CLASSES):
        """Create a d-input, q-class softmax regression model."""
        super().__init__()
        self.linear = nn.Linear(d, q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits for a batch of flattened images."""
        return self.linear(x)
