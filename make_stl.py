import torch 
import torchvision.transforms as transforms
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
import os
import torchvision
import tqdm
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import random
## read images from STL10 dataset and save them in a folder

def save_stl10_images(data_dir, save_dir):
    batch_size=128
    num_workers=4
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    ds = STL10(root=data_dir, split='unlabeled', download=True, transform=transform)
    dl_prev = DataLoader(ds, batch_size=100000, shuffle=False, num_workers=1)

    dataset = next(iter(dl_prev))

    print(f"Dataset size: {len(dataset[0])} images")

    # make dataset from tensor
    dataset = TensorDataset(dataset[0], dataset[1])

    # make dataloader
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    idx = 0
    for images, _ in tqdm.tqdm(dl, desc="Saving STL10 images"):
        for b in range(images.size(0)):
            img_path = os.path.join(save_dir, f'image_{idx:08d}.png')
            torchvision.utils.save_image(images[b].clamp(0, 1), img_path)  # expects CHW tensor
            idx += 1

if __name__ == "__main__":
    data_dir = './data_stl'
    save_dir = './datasets/stl/img'
    save_stl10_images(data_dir, save_dir)
    print(f"Images saved to {save_dir}")