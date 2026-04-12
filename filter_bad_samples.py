"""
Filter generated samples by detecting waffle/checkerboard periodic artifacts.
Uses FFT to find images with abnormally strong periodic patterns.

Usage:
    python filter_bad_samples.py --input_dir ./path/to/fid/folder --output_dir ./bad_samples --threshold 0.15
"""

import argparse
import os
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm


def periodic_artifact_score(img_path):
    """
    Compute a score indicating how much periodic/waffle artifact is present.
    Higher score = more artifact.

    Uses 2D FFT: periodic artifacts create strong off-center peaks in frequency space.
    Score = energy in mid-frequency band / total energy (excluding DC).
    """
    img = Image.open(img_path).convert('L')  # grayscale
    arr = np.array(img, dtype=np.float32)

    # 2D FFT and shift DC to center
    f = np.fft.fft2(arr)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    # Zero out DC component (center 5x5)
    dc_mask = np.zeros_like(magnitude)
    dc_mask[cy-2:cy+3, cx-2:cx+3] = 1
    magnitude_no_dc = magnitude * (1 - dc_mask)

    total_energy = magnitude_no_dc.sum() + 1e-8

    # Mid-frequency ring: where waffle artifacts typically appear
    # For 256x256 images, waffle patterns are usually at period 8-32px
    # corresponding to frequencies h/32 to h/8 from center
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((Y - cy)**2 + (X - cx)**2)

    r_min = min(h, w) // 32
    r_max = min(h, w) // 4
    mid_freq_mask = (dist_from_center >= r_min) & (dist_from_center <= r_max)

    mid_freq_energy = (magnitude_no_dc * mid_freq_mask).sum()

    # Peakiness: ratio of max peak in mid-freq band to mean
    mid_freq_vals = magnitude_no_dc[mid_freq_mask]
    peakiness = mid_freq_vals.max() / (mid_freq_vals.mean() + 1e-8)

    # Combine: high mid-freq energy fraction + high peakiness = artifact
    energy_frac = mid_freq_energy / total_energy
    score = energy_frac * np.log1p(peakiness / 100.0)

    return float(score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Folder with generated images')
    parser.add_argument('--output_dir', type=str, required=True, help='Folder to copy bad images into')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Score threshold above which images are flagged as bad. '
                             'If not set, uses mean + 2*std (auto).')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Instead of threshold, flag the worst top-K images.')
    args = parser.parse_args()

    image_files = [
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    image_files.sort()

    if not image_files:
        print(f"No images found in {args.input_dir}")
        return

    print(f"Scoring {len(image_files)} images...")
    scores = []
    for fname in tqdm(image_files):
        path = os.path.join(args.input_dir, fname)
        score = periodic_artifact_score(path)
        scores.append((fname, score))

    score_vals = np.array([s for _, s in scores])
    print(f"Score stats — min: {score_vals.min():.4f}, max: {score_vals.max():.4f}, "
          f"mean: {score_vals.mean():.4f}, std: {score_vals.std():.4f}")

    if args.top_k is not None:
        threshold = sorted(score_vals)[-args.top_k]
        print(f"Using top-{args.top_k} threshold: {threshold:.4f}")
    elif args.threshold is not None:
        threshold = args.threshold
        print(f"Using manual threshold: {threshold:.4f}")
    else:
        threshold = score_vals.mean() + 2 * score_vals.std()
        print(f"Using auto threshold (mean + 2*std): {threshold:.4f}")

    bad = [(fname, score) for fname, score in scores if score >= threshold]
    good = [(fname, score) for fname, score in scores if score < threshold]
    print(f"Found {len(bad)} bad images ({100*len(bad)/len(scores):.1f}%), {len(good)} good images")

    bad_dir = os.path.join(args.output_dir, 'bad')
    good_dir = os.path.join(args.output_dir, 'good')
    os.makedirs(bad_dir, exist_ok=True)
    os.makedirs(good_dir, exist_ok=True)

    for fname, score in bad:
        src = os.path.join(args.input_dir, fname)
        shutil.copy2(src, os.path.join(bad_dir, f"score{score:.4f}_{fname}"))

    for fname, score in good:
        src = os.path.join(args.input_dir, fname)
        shutil.copy2(src, os.path.join(good_dir, fname))

    print(f"Copied bad images to {bad_dir}")
    print(f"Copied good images to {good_dir}")

    # Print worst 20
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    print("\nWorst 20 images:")
    for fname, score in scores_sorted[:20]:
        print(f"  {score:.4f}  {fname}")


if __name__ == '__main__':
    main()
