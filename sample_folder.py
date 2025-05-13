import os
import random
import shutil
from pathlib import Path
import argparse

def sample_images(input_folder, output_folder, n):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    image_paths = list(input_folder.glob("*"))
    image_paths = [p for p in image_paths if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]

    if n > len(image_paths):
        raise ValueError(f"Requested {n} images, but only {len(image_paths)} available.")

    sampled = random.sample(image_paths, n)

    for i, path in enumerate(sampled):
        shutil.copy(path, output_folder / f"{i:04d}{path.suffix}")

    print(f"Saved {n} images to {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample N images from a folder.")
    parser.add_argument("source", type=str, help="Path to source image folder")
    parser.add_argument("destination", type=str, help="Path to output folder")
    parser.add_argument("number", type=int, help="Number of images to sample")

    args = parser.parse_args()
    sample_images(args.source, args.destination, args.number)
