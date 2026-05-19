import argparse

from smreg.bench import bench_forward, bench_matmul, smoke_test, validate_correctness
from smreg.train import compare_training, train_pytorch, train_triton


def parse_args() -> argparse.Namespace:
    """Parse command-line options for training, validation, and benchmarks."""
    parser = argparse.ArgumentParser(description="Softmax regression with PyTorch and Triton kernels.")
    parser.add_argument(
        "--mode",
        choices=[
            "smoke",
            "train",
            "train-pytorch",
            "train-triton",
            "compare-training",
            "bench-matmul",
            "bench-forward",
            "validate",
        ],
        default="smoke",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument(
        "--subset",
        type=int,
        default=20000,
        help="Training-set cap for report experiments; use 0 for the full dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for benchmark CSV files and plots.",
    )
    return parser.parse_args()


def main() -> None:
    """Dispatch the selected project command."""
    args = parse_args()
    subset = None if args.subset == 0 else args.subset

    if args.mode == "smoke":
        smoke_test()
    elif args.mode == "validate":
        validate_correctness(args.batch_size)
    elif args.mode == "train-pytorch":
        result = train_pytorch(args.epochs, args.batch_size, args.lr, subset)
        print(result)
    elif args.mode == "train-triton":
        result = train_triton(args.epochs, args.batch_size, args.lr, subset)
        print(result)
    elif args.mode == "train":
        compare_training(args.epochs, args.batch_size, args.lr, subset, args.output_dir)
    elif args.mode == "compare-training":
        compare_training(args.epochs, args.batch_size, args.lr, subset, args.output_dir)
    elif args.mode == "bench-matmul":
        bench_matmul(args.output_dir)
    elif args.mode == "bench-forward":
        bench_forward(args.output_dir)


if __name__ == "__main__":
    main()
