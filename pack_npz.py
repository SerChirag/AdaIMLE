## Pack a folder of images (searched recursively) into a single .npz (arr_0, uint8 NHWC)
## in the format evaluation/evaluator.py expects.
##
## Usage:
##   python pack_npz.py --src <image folder> --out <file.npz> [--n 50000]

import argparse
import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tqdm
from PIL import Image, PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="folder of images (searched recursively)")
    parser.add_argument("--out", required=True, help="output .npz path")
    parser.add_argument("--n", type=int, default=50000, help="number of images to pack")
    parser.add_argument("--seed", type=int, default=42, help="seed for subsampling when the folder has more than --n images")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    files = sorted(
        os.path.join(root, f)
        for root, _, names in os.walk(args.src)
        for f in names
        if os.path.splitext(f.lower())[1] in EXTS
    )
    if len(files) < args.n:
        raise ValueError(f"{args.src} has only {len(files)} images, need {args.n}")
    if len(files) > args.n:
        random.Random(args.seed).shuffle(files)
        files = files[:args.n]

    first = load_rgb(files[0])
    arr = np.empty((len(files),) + first.shape, dtype=np.uint8)
    with ThreadPoolExecutor(args.workers) as pool:
        for i, img in enumerate(tqdm.tqdm(pool.map(load_rgb, files), total=len(files), desc=f"packing {args.src}")):
            if img.shape != first.shape:
                raise ValueError(f"{files[i]} has shape {img.shape}, expected {first.shape}")
            arr[i] = img

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, arr_0=arr)
    print(f"Saved {arr.shape} to {args.out}")


if __name__ == "__main__":
    main()
