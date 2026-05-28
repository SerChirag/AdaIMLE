"""
Sample X images per class folder from a source imagenet-style directory.

Usage:
    python sample_imagenet.py --src /imagenet_pr_50k --n 10 --dst /imagenet_pr_10k
    python sample_imagenet.py --src /imagenet_pr_50k --n 1  --dst /imagenet_pr_1k
"""

import argparse
import os
import random
import shutil
from pathlib import Path


def sample_folders(src: Path, dst: Path, n: int, seed: int = 42):
    rng = random.Random(seed)

    class_dirs = sorted([d for d in src.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No subdirectories found in {src}")

    print(f"Found {len(class_dirs)} class folders. Sampling {n} images each → {dst}")

    for class_dir in class_dirs:
        images = sorted([
            f for f in class_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        ])

        if len(images) < n:
            print(f"  WARNING: {class_dir.name} has only {len(images)} images (requested {n}), taking all")
            sampled = images
        else:
            sampled = rng.sample(images, n)

        out_dir = dst / class_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for img in sampled:
            shutil.copy2(img, out_dir / img.name)

    total = n * len(class_dirs)
    print(f"Done. ~{total} images saved to {dst}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source folder (e.g. /imagenet_pr_50k)")
    parser.add_argument("--n", type=int, required=True, help="Images to sample per class")
    parser.add_argument("--dst", required=True, help="Destination folder (e.g. /imagenet_pr_10k)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        raise FileNotFoundError(f"Source folder not found: {src}")
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}  (delete it first or choose a new path)")

    sample_folders(src, dst, args.n, args.seed)


if __name__ == "__main__":
    main()
