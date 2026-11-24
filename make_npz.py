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

    # Preallocate from the first image's shape: for 30k 256x256x3 images the
    # list + np.stack route peaks at twice the final array (~12 GB).
    arr = None
    shape = None
    for i, path in enumerate(tqdm.tqdm(files, desc=f"Loading {len(files)} images")):
        a = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        if arr is None:
            shape = a.shape
            arr = np.empty((len(files),) + shape, dtype=np.uint8)
        elif a.shape != shape:
            raise ValueError(f"{path} has shape {a.shape}, expected {shape}")
        arr[i] = a

    return arr

jobs = [
    # ('/localscratch/cva19/celeba_comb', '/localscratch/cva19/eval_refs/celeba_ref_30000.npz', 30000),
    # ('/localscratch/cva19/shortcut_results/celebd/128/samples',
    #  '/localscratch/cva19/eval_refs/celebd_128_samples_49984.npz', 49984),
    ('/localscratch/cva19/shortcut_results/celebd/1/samples',
     '/localscratch/cva19/eval_refs/celebd_1_samples_49984.npz', 49984),
]

for folder, out_path, n in jobs:
    if os.path.exists(out_path):
        print(f'Skipping {out_path} (exists)')
        continue
    array = create_npz_from_folder(folder, n)
    np.savez(out_path, arr_0=array)
    print(f'Saved {array.shape} to {out_path}')
    del array
