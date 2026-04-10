## Script to create .npz files from a folder of images. Each .npz file will contain a specified number of randomly sampled images from the folder, stored as a NumPy array. 
## The script uses a fixed random seed for reproducibility.
## This create .npz files used by evaluator.py

import os
import numpy as np
from PIL import Image
import random
import tqdm as tqdm

def create_npz_from_folder(folder, n_samples, seed=42):
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f.lower())[1] in exts
    ]
    rng = random.Random(seed)
    rng.shuffle(files)
    files = files[:n_samples]

    imgs = []

    ## Add tqdm progress bar to show the progress of loading images
    for path in tqdm.tqdm(files, desc=f"Loading {n_samples} images"):
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
        imgs.append(arr)

    return np.stack(imgs, axis=0)

subset = [50000]

for n in subset:
    out_path = f'/localscratch/cva19/latent_vamp/evaluation/imagenet_novamp.npz'
    array = create_npz_from_folder('/localscratch/cva19/latent_vamp/new-vanilla-results/imagenet256-mclure-mark2-novamp-embmult10x/train/fid', n)
    np.savez(out_path, arr_0=array)
    print(f'Saved {n} samples to {out_path}')
    del array
