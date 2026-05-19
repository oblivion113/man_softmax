import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bench import print_markdown_table, save_bar_plot, save_csv
from .config import INPUT_DIM, NUM_CLASSES
from .data import load_fashion_mnist
from .model import SoftmaxRegression
from .triton_ops import (
    cuda_device,
    triton_bias_add,
    triton_fused_softmax_ce,
    triton_matmul,
    triton_row_sum,
    triton_sgd_update,
    triton_softmax_ce_backward,
)


@dataclass
class TrainResult:
    """Summary statistics for a completed training run."""

    train_loss: float
    test_accuracy: float
    elapsed_seconds: float


def evaluate_pytorch(model: nn.Module, loader, device: torch.device) -> float:
    """Measure classification accuracy for the PyTorch baseline model."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    model.train()
    return correct / max(total, 1)


def train_pytorch(epochs: int, batch_size: int, lr: float, subset: int | None) -> TrainResult:
    """Train the reference PyTorch softmax regression model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_fashion_mnist(batch_size, subset)
    torch.manual_seed(0)
    model = SoftmaxRegression().to(device)
    nn.init.normal_(model.linear.weight, mean=0.0, std=0.01)
    nn.init.zeros_(model.linear.bias)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    start = time.perf_counter()
    last_loss = 0.0
    for epoch in range(epochs):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            last_loss = loss.item()
        print(f"PyTorch epoch {epoch + 1}: loss={last_loss:.4f}")

    elapsed = time.perf_counter() - start
    acc = evaluate_pytorch(model, test_loader, device)
    return TrainResult(last_loss, acc, elapsed)


def train_triton(epochs: int, batch_size: int, lr: float, subset: int | None) -> TrainResult:
    """Train softmax regression using explicit Triton forward/backward kernels."""
    device = cuda_device()
    train_loader, test_loader = load_fashion_mnist(batch_size, subset)

    torch.manual_seed(0)
    W = torch.empty((INPUT_DIM, NUM_CLASSES), device=device, dtype=torch.float32)
    b = torch.zeros((NUM_CLASSES,), device=device, dtype=torch.float32)
    nn.init.normal_(W, mean=0.0, std=0.01)

    start = time.perf_counter()
    last_loss = 0.0
    for epoch in range(epochs):
        for x, y in train_loader:
            x = x.to(device, dtype=torch.float32).contiguous()
            y = y.to(device).contiguous()

            logits = triton_bias_add(triton_matmul(x, W), b)
            losses, probs = triton_fused_softmax_ce(logits, y)
            dO = triton_softmax_ce_backward(probs, y)
            dW = triton_matmul(x.t().contiguous(), dO)
            db = triton_row_sum(dO)
            triton_sgd_update(W, dW, lr)
            triton_sgd_update(b, db, lr)
            last_loss = losses.mean().item()
        print(f"Triton epoch {epoch + 1}: loss={last_loss:.4f}")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    acc = evaluate_triton(W, b, test_loader, device)
    return TrainResult(last_loss, acc, elapsed)


def evaluate_triton(W: torch.Tensor, b: torch.Tensor, loader, device: torch.device) -> float:
    """Measure classification accuracy for weights managed by the Triton loop."""
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, dtype=torch.float32).contiguous()
        y = y.to(device)
        logits = triton_bias_add(triton_matmul(x, W), b)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def compare_training(
    epochs: int,
    batch_size: int,
    lr: float,
    subset: int | None,
    output_dir: str = "results",
) -> list[dict]:
    """Train both implementations and save a compact performance comparison.

    The comparison uses matching parameter initialization: W ~ N(0, 0.01) and
    b = 0 for both PyTorch and Triton. This makes final loss/accuracy differences
    mainly reflect implementation and data-order effects rather than different
    starting weights.
    """
    print("Running PyTorch baseline...")
    torch_result = train_pytorch(epochs, batch_size, lr, subset)
    print("Running Triton implementation...")
    triton_result = train_triton(epochs, batch_size, lr, subset)

    rows = [
        {
            "implementation": "PyTorch",
            "train_loss": torch_result.train_loss,
            "test_accuracy": torch_result.test_accuracy,
            "elapsed_seconds": torch_result.elapsed_seconds,
        },
        {
            "implementation": "Triton",
            "train_loss": triton_result.train_loss,
            "test_accuracy": triton_result.test_accuracy,
            "elapsed_seconds": triton_result.elapsed_seconds,
        },
    ]
    headers = ["implementation", "train_loss", "test_accuracy", "elapsed_seconds"]
    print_markdown_table(headers, rows)

    output_path = Path(output_dir)
    save_csv(output_path / "training_comparison.csv", headers, rows)
    save_bar_plot(
        output_path / "training_comparison.png",
        rows,
        "implementation",
        ["train_loss", "test_accuracy", "elapsed_seconds"],
        "Training Comparison",
    )
    return rows
