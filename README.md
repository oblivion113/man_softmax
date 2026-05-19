# Softmax Regression: Triton vs PyTorch

This project implements softmax regression for FashionMNIST with both a normal
PyTorch baseline and explicit Triton kernels. It is structured for a short
course report about custom GPU kernels, operator fusion, and performance
comparison.

## Environment

Use the course conda environment:

```bash
conda activate AICourse
```

Expected core packages:

```bash
python -c "import torch, triton; print(torch.__version__); print(triton.__version__); print(torch.cuda.is_available())"
```

On the target laptop this should report CUDA as available.

## Project Structure

```text
softmax_regression.py     CLI entry point
smreg/
  bench.py                Smoke tests, correctness validation, benchmarks, CSV/plot helpers
  config.py               Shared constants such as image dimension and class count
  data.py                 FashionMNIST DataLoader setup
  model.py                PyTorch softmax regression baseline
  train.py                PyTorch and Triton training loops
  triton_ops.py           Triton kernels and Python wrapper functions
refs/                     Original Triton tutorial reference files
```

## Quick Checks

Run a smoke test first:

```bash
python softmax_regression.py --mode smoke
```

Run a stricter Triton-vs-PyTorch tensor comparison:

```bash
python softmax_regression.py --mode validate --batch-size 256
```

This checks the Triton forward pass, fused softmax/cross-entropy loss, and
backward gradients against PyTorch autograd.

## Training

Train only the PyTorch baseline:

```bash
python softmax_regression.py --mode train-pytorch --epochs 3 --batch-size 256
```

Train only the Triton implementation:

```bash
python softmax_regression.py --mode train-triton --epochs 3 --batch-size 256
```

Train both implementations for a direct comparison:

```bash
python softmax_regression.py --mode train --epochs 8 --batch-size 256 --subset 20000
```

This prints a Markdown table and writes:

```text
results/training_comparison.csv
results/training_comparison.png
```

You can also call the same comparison explicitly:

```bash
python softmax_regression.py --mode compare-training --epochs 8 --batch-size 256 --subset 20000 --output-dir results
```

By default, training uses a 20000-sample subset so the reported accuracy is more
convincing while still running quickly. Use the full dataset with:

```bash
python softmax_regression.py --mode train --subset 0
```

## Benchmarks and Report Artifacts

Forward-pass benchmark:

```bash
python softmax_regression.py --mode bench-forward --output-dir results
```

Matmul benchmark:

```bash
python softmax_regression.py --mode bench-matmul --output-dir results
```

Each benchmark prints a Markdown table and writes:

```text
results/forward_benchmark.csv
results/forward_benchmark.png
results/matmul_benchmark.csv
results/matmul_benchmark.png
```

If `matplotlib` is not installed, the CSV and terminal table are still produced
and only PNG generation is skipped.

## Git

The project is initialized as a Git repository on the `main` branch. Check the
current working tree with:

```bash
git status
```
