import gzip
import hashlib
import shutil
import subprocess

import torch

from .config import DATA_DIR


FASHION_MNIST_MIRRORS = [
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
]


def _md5(path) -> str:
    """Return the hex MD5 digest for a local file."""
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_with_curl(url: str, destination) -> None:
    """Download a file with curl retries.

    torchvision's FashionMNIST downloader can leave truncated files on some
    networks. curl reports short responses clearly and retries transient
    failures, which makes the course demo less fragile.
    """
    subprocess.run(
        [
            "curl",
            "-L",
            "--retry",
            "5",
            "--retry-delay",
            "1",
            "--fail",
            "--output",
            str(destination),
            url,
        ],
        check=True,
    )


def _ensure_fashion_mnist_archives(resources: list[tuple[str, str]]) -> None:
    """Download and extract FashionMNIST raw archives if they are missing or corrupt."""
    raw_dir = DATA_DIR / "FashionMNIST" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for filename, expected_md5 in resources:
        archive_path = raw_dir / filename
        extracted_path = raw_dir / filename.removesuffix(".gz")
        if archive_path.exists() and _md5(archive_path) == expected_md5:
            if not extracted_path.exists():
                with gzip.open(archive_path, "rb") as src, extracted_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            continue

        if archive_path.exists():
            archive_path.unlink()
        _download_with_curl(FASHION_MNIST_MIRRORS[0] + filename, archive_path)
        actual_md5 = _md5(archive_path)
        if actual_md5 != expected_md5:
            archive_path.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded {filename} with MD5 {actual_md5}, expected {expected_md5}.")

        with gzip.open(archive_path, "rb") as src, extracted_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)


def load_fashion_mnist(batch_size: int, subset: int | None = None):
    """Return FashionMNIST train/test loaders with flattened 784-feature images.

    Args:
        batch_size: Number of samples per mini-batch.
        subset: Optional cap for the training set. A proportional test subset is
            used to keep quick experiments fast.
    """
    from torchvision import datasets, transforms

    # torchvision 0.23 ships FashionMNIST with only the old S3 HTTP mirror.
    # Keeping the official GitHub raw mirror first makes downloads less brittle.
    datasets.FashionMNIST.mirrors = FASHION_MNIST_MIRRORS
    _ensure_fashion_mnist_archives(datasets.FashionMNIST.resources)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    train_set = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=transform)

    if subset is not None:
        train_set = torch.utils.data.Subset(train_set, range(min(subset, len(train_set))))
        test_size = min(max(subset // 5, batch_size), len(test_set))
        test_set = torch.utils.data.Subset(test_set, range(test_size))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
