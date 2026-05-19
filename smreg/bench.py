import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import INPUT_DIM, NUM_CLASSES
from .model import SoftmaxRegression
from .triton_ops import (
    TRITON_IMPORT_ERROR,
    cuda_device,
    triton,
    triton_bias_add,
    triton_fused_softmax_ce,
    triton_matmul,
    triton_row_sum,
    triton_softmax_ce_backward,
    triton_unfused_softmax_ce,
)


def _format_value(value) -> str:
    """Format benchmark values for compact Markdown table display."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_markdown_table(headers: list[str], rows: list[dict]) -> None:
    """Print benchmark rows as a Markdown table for direct report reuse."""
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(_format_value(row[h]) for h in headers) + " |")


def save_csv(path: Path, headers: list[str], rows: list[dict]) -> None:
    """Write benchmark rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def save_line_plot(path: Path, rows: list[dict], x_key: str, y_keys: list[str], title: str, ylabel: str) -> None:
    """Save a simple line plot if matplotlib is installed."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib is not installed; skipped plot {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    xs = [row[x_key] for row in rows]
    for y_key in y_keys:
        plt.plot(xs, [row[y_key] for row in rows], marker="o", label=y_key)
    plt.xlabel(x_key)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_bar_plot(path: Path, rows: list[dict], x_key: str, y_keys: list[str], title: str) -> None:
    """Save polished small-multiple bar charts for categorical comparisons."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print(f"matplotlib or numpy is not installed; skipped plot {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row[x_key]) for row in rows]
    x = np.arange(len(labels))
    colors = ["#2563eb", "#f97316", "#16a34a", "#7c3aed"]
    metric_titles = {
        "train_loss": "Final Training Loss",
        "test_accuracy": "Test Accuracy",
        "elapsed_seconds": "Training Time",
    }
    metric_units = {
        "train_loss": "loss",
        "test_accuracy": "accuracy",
        "elapsed_seconds": "seconds",
    }

    fig, axes = plt.subplots(1, len(y_keys), figsize=(3.4 * len(y_keys), 3.8))
    if len(y_keys) == 1:
        axes = [axes]

    for ax, y_key in zip(axes, y_keys):
        values = [row[y_key] for row in rows]
        bars = ax.bar(x, values, width=0.34, color=colors[:len(values)], edgecolor="none")
        ax.set_title(metric_titles.get(y_key, y_key.replace("_", " ").title()), fontsize=12, pad=12)
        ax.set_ylabel(metric_units.get(y_key, "value"), fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.grid(True, axis="y", alpha=0.18, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#d4d4d8")
        ax.spines["bottom"].set_color("#d4d4d8")

        ymax = max(values) if values else 1.0
        ymin = min(values) if values else 0.0
        if y_key == "test_accuracy":
            ax.set_ylim(0, min(1.0, ymax * 1.16))
        elif ymin >= 0:
            ax.set_ylim(0, ymax * 1.18)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                _format_value(value),
                ha="center",
                va="bottom",
                fontsize=10,
                color="#18181b",
                fontweight="medium",
            )

    fig.suptitle(title, fontsize=15, fontweight="semibold", y=0.98)
    fig.patch.set_facecolor("white")
    fig.tight_layout(rect=[0, 0, 1, 0.90], w_pad=1.4)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def smoke_test() -> None:
    """Run a quick environment and correctness check for the current machine."""
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if triton is None:
        print("Triton import failed:", TRITON_IMPORT_ERROR)
    else:
        print("Triton:", triton.__version__)

    if not torch.cuda.is_available():
        print("Skipping Triton runtime checks because no CUDA device is visible.")
        x = torch.randn(4, INPUT_DIM)
        y = torch.tensor([0, 1, 2, 3])
        model = SoftmaxRegression()
        loss = F.cross_entropy(model(x), y)
        print(f"CPU PyTorch smoke loss: {loss.item():.4f}")
        return

    device = cuda_device()
    torch.manual_seed(0)
    x = torch.randn((16, INPUT_DIM), device=device, dtype=torch.float32)
    W = torch.randn((INPUT_DIM, NUM_CLASSES), device=device, dtype=torch.float32) * 0.01
    b = torch.randn((NUM_CLASSES,), device=device, dtype=torch.float32) * 0.01
    labels = torch.randint(0, NUM_CLASSES, (16,), device=device)

    logits_triton = triton_bias_add(triton_matmul(x, W), b)
    logits_torch = x @ W + b
    losses, probs = triton_fused_softmax_ce(logits_triton, labels)
    torch_loss = F.cross_entropy(logits_torch, labels, reduction="none")
    assert torch.allclose(logits_triton, logits_torch, atol=1e-3, rtol=1e-3)
    assert torch.allclose(losses, torch_loss, atol=1e-3, rtol=1e-3)
    assert torch.allclose(probs, torch.softmax(logits_torch, dim=1), atol=1e-3, rtol=1e-3)
    print("Triton smoke checks passed.")


def validate_correctness(batch_size: int) -> None:
    """Compare Triton forward/backward tensors against PyTorch autograd."""
    device = cuda_device()
    torch.manual_seed(0)
    x = torch.randn((batch_size, INPUT_DIM), device=device, dtype=torch.float32)
    labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
    W = torch.randn((INPUT_DIM, NUM_CLASSES), device=device, dtype=torch.float32) * 0.01
    b = torch.randn((NUM_CLASSES,), device=device, dtype=torch.float32) * 0.01

    logits = triton_bias_add(triton_matmul(x, W), b)
    losses, probs = triton_fused_softmax_ce(logits, labels)
    dO = triton_softmax_ce_backward(probs, labels)
    dW = triton_matmul(x.t().contiguous(), dO)
    db = triton_row_sum(dO)

    x_ref = x.detach().clone().requires_grad_(True)
    W_ref = W.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    loss_ref = F.cross_entropy(x_ref @ W_ref + b_ref, labels)
    loss_ref.backward()

    assert torch.allclose(losses.mean(), loss_ref, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dW, W_ref.grad, atol=1e-3, rtol=1e-3)
    assert torch.allclose(db, b_ref.grad, atol=1e-3, rtol=1e-3)
    print("Forward and backward correctness checks passed.")


def bench_matmul(output_dir: str = "results") -> list[dict]:
    """Benchmark grouped Triton matmul, naive Triton matmul, and torch.matmul.

    Args:
        output_dir: Directory where CSV and PNG artifacts are saved.

    Returns:
        A list of dictionaries containing one row per matrix size.
    """
    device = cuda_device()
    rows = []
    for n in [256, 512, 1024, 2048, 4096]:
        a = torch.randn((n, n), device=device, dtype=torch.float32)
        b = torch.randn((n, n), device=device, dtype=torch.float32)
        torch_ms = triton.testing.do_bench(lambda: torch.matmul(a, b))
        grouped_ms = triton.testing.do_bench(lambda: triton_matmul(a, b, grouped=True))
        naive_ms = triton.testing.do_bench(lambda: triton_matmul(a, b, grouped=False))
        rows.append({
            "size": n,
            "torch_ms": torch_ms,
            "triton_grouped_ms": grouped_ms,
            "triton_naive_ms": naive_ms,
        })

    headers = ["size", "torch_ms", "triton_grouped_ms", "triton_naive_ms"]
    print_markdown_table(headers, rows)
    output_path = Path(output_dir)
    save_csv(output_path / "matmul_benchmark.csv", headers, rows)
    save_line_plot(
        output_path / "matmul_benchmark.png",
        rows,
        "size",
        ["torch_ms", "triton_grouped_ms", "triton_naive_ms"],
        "Matmul Benchmark",
        "milliseconds",
    )
    return rows


def bench_forward(output_dir: str = "results") -> list[dict]:
    """Benchmark the training forward pass with fused and unfused CE kernels.

    Args:
        output_dir: Directory where CSV and PNG artifacts are saved.

    Returns:
        A list of dictionaries containing one row per batch size.
    """
    device = cuda_device()
    W = torch.randn((INPUT_DIM, NUM_CLASSES), device=device, dtype=torch.float32) * 0.01
    b = torch.zeros((NUM_CLASSES,), device=device, dtype=torch.float32)
    rows = []
    for batch in [64, 128, 256, 512, 1024, 2048, 4096]:
        x = torch.randn((batch, INPUT_DIM), device=device, dtype=torch.float32)
        labels = torch.randint(0, NUM_CLASSES, (batch,), device=device)

        def fused():
            logits = triton_bias_add(triton_matmul(x, W), b)
            return triton_fused_softmax_ce(logits, labels)

        def unfused():
            logits = triton_bias_add(triton_matmul(x, W), b)
            return triton_unfused_softmax_ce(logits, labels)

        torch_ms = triton.testing.do_bench(lambda: F.cross_entropy(x @ W + b, labels))
        fused_ms = triton.testing.do_bench(fused)
        unfused_ms = triton.testing.do_bench(unfused)
        rows.append({
            "batch": batch,
            "torch_ms": torch_ms,
            "triton_fused_ms": fused_ms,
            "triton_unfused_ms": unfused_ms,
        })

    headers = ["batch", "torch_ms", "triton_fused_ms", "triton_unfused_ms"]
    print_markdown_table(headers, rows)
    output_path = Path(output_dir)
    save_csv(output_path / "forward_benchmark.csv", headers, rows)
    save_line_plot(
        output_path / "forward_benchmark.png",
        rows,
        "batch",
        ["torch_ms", "triton_fused_ms", "triton_unfused_ms"],
        "Forward Pass Benchmark",
        "milliseconds",
    )
    return rows
