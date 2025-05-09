import os
import argparse
import random
from PIL import Image
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help='Folder with input images')
    parser.add_argument('--grid', type=str, default='4x4', help='Grid size in NxM format (e.g., 3x3)')
    parser.add_argument('--out', type=str, default='output/grid.jpg', help='Output path for grid image')
    parser.add_argument('--size', type=int, default=256, help='Size to resize each image (default: 256x256)')
    return parser.parse_args()

def load_images(folder, num_images, size):
    all_images = [os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))]
    chosen = random.sample(all_images, min(num_images, len(all_images)))
    images = [Image.open(img_path).convert('RGB').resize((size, size)) for img_path in chosen]
    return images

def make_grid(images, nrow, ncol, size):
    grid_img = Image.new('RGB', (size * ncol, size * nrow))
    for idx, img in enumerate(images):
        row = idx // ncol
        col = idx % ncol
        grid_img.paste(img, (col * size, row * size))
    return grid_img

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    try:
        nrow, ncol = map(int, args.grid.lower().split('x'))
    except:
        raise ValueError("Grid format must be NxM (e.g., 3x4)")
    
    images = load_images(args.folder, nrow * ncol, args.size)
    if len(images) < nrow * ncol:
        raise ValueError(f"Not enough images in folder ({len(images)} found, need {nrow * ncol})")
    
    grid_img = make_grid(images, nrow, ncol, args.size)
    grid_img.save(args.out)
    print(f"Saved grid image to {args.out}")

if __name__ == "__main__":
    main()
