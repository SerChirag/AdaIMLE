import numpy as np
import pickle
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from PIL import Image, PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024  # raise limit to handle large ICC profiles in PNGs
from datasets import load_dataset
from torch.utils.data import Dataset

from helpers.utils import get_world_size
from models import parse_layer_string
from torchvision.datasets import CIFAR10, STL10
from helpers.autoencoder import load_autoencoder, encode_images_to_latents
from helpers.cache_utils import image_cache_key, load_image_cache, save_image_cache


def set_up_data(H):

    blocks = parse_layer_string(H.dec_blocks)
    H.block_res = [s[0] for s in blocks]
    H.res = sorted(set([s[0] for s in blocks if s[0] <= H.max_hierarchy]))
    H.latent_spatial_size = max(H.block_res)

    # trX: image array (NHWC uint8 or similar); trY: label array or None
    trY = None

    if H.dataset == 'imagenet32':
        trX, trY, teX = imagenet32(H.data_root)
        H.image_size = 32
        H.image_channels = 3
    elif H.dataset == 'imagenet_folder':
        trX, trY = None, None  # loaded below via cache path
        H.image_channels = 3
    elif H.dataset in ['fewshot', 'fewshot512', 'fewshot64']:
        trX, vaX, teX = few_shot_image_folder(H.data_root, H.image_size)
        H.image_channels = 3
    elif H.dataset == 'imagenet64':
        trX, vaX, teX = imagenet64(H.data_root)
        H.image_size = 64
        H.image_channels = 3
    elif H.dataset == 'ffhq_256':
        trX, vaX, teX = ffhq256(H.data_root)
        H.image_size = 256
        H.image_channels = 3
    elif H.dataset == 'ffhq_1024':
        trX, vaX, teX = ffhq1024(H.data_root)
        H.image_size = 1024
        H.image_channels = 3
    elif H.dataset == 'cifar10':
        (trX, trY_raw), (teX, _) = cifar10(H.data_root, one_hot=False)
        trY = trY_raw.reshape(-1)
        H.image_size = 32
        H.image_channels = 3
    elif H.dataset == "stl10":
        trX, vaX, teX = stl10(H.data_root)
        H.image_size = 64
        H.image_channels = 3
    elif H.dataset == 'lsun':
        trX, vaX, teX = lsun_church(H.data_root)
        H.image_size     = 256
        H.image_channels = 3
    else:
        raise ValueError('unknown dataset: ', H.dataset)

    device = torch.device("cuda", torch.cuda.current_device())

    autoencoder = load_autoencoder(H, device)
    latent_probe = encode_images_to_latents(
        autoencoder,
        torch.zeros(1, 3, H.image_size, H.image_size, device=device),
        target_spatial=(H.latent_spatial_size, H.latent_spatial_size),
    )
    H.image_channels = latent_probe.shape[1]

    train_len = None
    use_cache = bool(getattr(H, 'use_cache', True))
    cache_dir = getattr(H, 'cache_dir', './cache')
    num_classes = getattr(H, 'num_classes', 0)

    if H.dataset == 'stl10':
        cached = None
        if use_cache:
            key = image_cache_key(H.data_root, H.image_size, H.dataset,
                                   cache_dataset_id=getattr(H, 'cache_dataset_id', ''))
            cached = load_image_cache(cache_dir, key, expected_size=len(trX))
        if cached is not None:
            # legacy: plain tensor
            tensor = cached if isinstance(cached, torch.Tensor) else cached['images']
            print(f"[cache] Loaded image tensor from cache ({tensor.shape[0]} images).")
            train_data = TensorDataset(tensor)
        else:
            chunks = []
            for batch in DataLoader(trX, batch_size=2048):
                chunks.append(((batch[0] + 1) * 127.5).clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1))
            full_tensor = torch.cat(chunks, dim=0)
            del chunks
            train_data = TensorDataset(full_tensor)
            if use_cache:
                save_image_cache(cache_dir, key, full_tensor)
        valid_data = train_data
        untranspose = False
        train_len = len(train_data)

    elif H.dataset == 'lsun':
        train_data = trX
        valid_data = trX
        train_len = train_data.ds.num_rows
        untranspose = True

    elif H.dataset == 'imagenet32':
        imgs = torch.as_tensor(trX).permute(0, 2, 3, 1)   # [N, H, W, 3]
        if num_classes > 0 and trY is not None:
            labels_t = torch.as_tensor(trY, dtype=torch.long)
            sort_idx = torch.argsort(labels_t, stable=True)
            imgs = imgs[sort_idx]
            labels_t = labels_t[sort_idx]
            H.labels = labels_t
            train_data = TensorDataset(imgs, labels_t)
        else:
            H.labels = None
            train_data = TensorDataset(imgs)
        valid_data = None
        train_len = len(train_data)
        untranspose = False

    elif H.dataset == 'cifar10':
        imgs = torch.as_tensor(trX)   # already [N, H, W, 3]
        if num_classes > 0 and trY is not None:
            labels_t = torch.as_tensor(trY, dtype=torch.long)
            sort_idx = torch.argsort(labels_t, stable=True)
            imgs = imgs[sort_idx]
            labels_t = labels_t[sort_idx]
            H.labels = labels_t
            train_data = TensorDataset(imgs, labels_t)
        else:
            H.labels = None
            train_data = TensorDataset(imgs)
        valid_data = None
        train_len = len(train_data)
        untranspose = False

    elif H.dataset == 'imagenet_folder':
        # Load via ImageFolder with caching.  Data is sorted by class
        # (ImageFolder iterates in folder-alphabetical order).
        key = image_cache_key(H.data_root, H.image_size, H.dataset, sorted_by_class=True,
                               cache_dataset_id=getattr(H, 'cache_dataset_id', ''))
        cached = load_image_cache(cache_dir, key) if use_cache else None
        if cached is not None and isinstance(cached, dict):
            imgs   = cached['images']
            labels_t = cached['labels']
            print(f"[cache] Loaded imagenet_folder from cache ({imgs.shape[0]} images).")
        else:
            imgs, labels_t = _load_imagefolder_to_tensors(H.data_root, H.image_size)
            if use_cache:
                save_image_cache(cache_dir, key, {'images': imgs, 'labels': labels_t})

        # Ensure class-sorted order (ImageFolder already is, but sort for safety)
        sort_idx = torch.argsort(labels_t, stable=True)
        imgs     = imgs[sort_idx]
        labels_t = labels_t[sort_idx]

        H.labels = labels_t
        train_data  = TensorDataset(imgs, labels_t)
        valid_data  = None
        train_len   = len(train_data)
        untranspose = False

    elif H.dataset not in ['fewshot', 'fewshot512', 'fewshot64']:
        train_data = TensorDataset(torch.as_tensor(trX))
        H.labels = None
        valid_data = None
        untranspose = False
        train_len = len(train_data)

    else:
        # fewshot / fewshot64 / fewshot512
        cached = None
        if use_cache:
            key = image_cache_key(H.data_root, H.image_size, H.dataset,
                                   cache_dataset_id=getattr(H, 'cache_dataset_id', ''))
            cached = load_image_cache(cache_dir, key, expected_size=len(trX))
        if cached is not None:
            tensor = cached if isinstance(cached, torch.Tensor) else cached['images']
            print(f"[cache] Loaded image tensor from cache ({tensor.shape[0]} images).")
            train_data = TensorDataset(tensor)
        else:
            chunks = []
            for batch in DataLoader(trX, batch_size=2048):
                chunks.append((batch[0] * 255).clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1))
            full_tensor = torch.cat(chunks, dim=0)
            del chunks
            train_data = TensorDataset(full_tensor)
            if use_cache:
                save_image_cache(cache_dir, key, full_tensor)
        H.labels = None
        valid_data = train_data
        untranspose = False
        train_len = len(train_data)


    H.global_batch_size = H.n_batch * get_world_size()
    effective_len = H.subset_len if H.subset_len != -1 else train_len
    H.train_len = effective_len
    H.total_iters = H.num_epochs * ((effective_len + H.global_batch_size - 1) // H.global_batch_size)

    if H.subset_len != -1:
        g = torch.Generator()
        g.manual_seed(H.seed)
        subset_indices = torch.randperm(train_len, generator=g)[:H.subset_len].tolist()
        train_data = Subset(train_data, subset_indices)

    def preprocess_func(x):
        nonlocal untranspose
        'takes in a data example and returns the preprocessed input'
        'as well as the input processed for the loss'
        if untranspose:
            x[0] = x[0].permute(0, 2, 3, 1)
        inp = x[0].to(device=device, non_blocking=True).float()
        inp.mul_(1./127.5).add_(-1)
        target = inp.permute(0, 3, 1, 2)
        target = encode_images_to_latents(
            autoencoder,
            target,
            target_spatial=(H.latent_spatial_size, H.latent_spatial_size),
        )
        target = target.permute(0, 2, 3, 1).contiguous()
        # Return 3-tuple: (pixel_input, labels_or_None, latent_target)
        # Callers should use preprocess_func(x)[-1] to get the latent target.
        if num_classes > 0 and len(x) > 1:
            labels_batch = x[1].to(device=device, non_blocking=True)
            return inp, labels_batch, target
        return inp, None, target

    return H, train_data, valid_data, preprocess_func, autoencoder


def _pil_loader(path: str) -> Image.Image:
    """PIL loader that handles PNGs with large ICC profiles (MAX_TEXT_CHUNK raised at import)."""
    with open(path, "rb") as f:
        img = Image.open(f)
        img.load()  # force decode before file closes
    return img.convert("RGB")


def _load_imagefolder_to_tensors(data_root, image_size):
    """Load an ImageFolder dataset into (images_NHWC_uint8, labels_int64) tensors.
    ImageFolder iterates in class-sorted folder order, so no extra sort is needed
    (though set_up_data sorts anyway for safety)."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # [0, 1] float, NCHW
    ])
    dataset = ImageFolder(data_root, transform=transform, loader=_pil_loader)
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=False)
    imgs_list, lbls_list = [], []
    for imgs, lbls in loader:
        imgs_list.append((imgs * 255).clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1))
        lbls_list.append(lbls)
    images = torch.cat(imgs_list, dim=0)   # [N, H, W, 3] uint8
    labels = torch.cat(lbls_list, dim=0)   # [N] int64
    return images, labels


def mkdir_p(path):
    os.makedirs(path, exist_ok=True)


def flatten(outer):
    return [el for inner in outer for el in inner]


def unpickle_cifar10(file):
    fo = open(file, 'rb')
    data = pickle.load(fo, encoding='bytes')
    fo.close()
    data = dict(zip([k.decode() for k in data.keys()], data.values()))
    return data


def few_shot_image_folder(data_root, image_size):
    transform_list = [
        transforms.Resize((int(image_size), int(image_size))),
        transforms.ToTensor(),
    ]
    trans = transforms.Compose(transform_list)
    train_data = ImageFolder(data_root, trans)
    return train_data, train_data, train_data


def imagenet32(data_root):

    files = sorted([f for f in os.listdir(data_root) if f.endswith(".npz")])

    images, labels = [], []

    for f in files:
        batch = np.load(os.path.join(data_root, f))
        X = batch["data"]        # shape (N, 3072)
        Y = batch["labels"]      # shape (N,)

        # reshape to (N, 3, 32, 32)
        X = X.reshape(-1, 3, 32, 32)
        images.append(X)
        labels.append(Y)

    images = np.concatenate(images)
    labels = np.concatenate(labels) - 1

    return images, labels, None


def imagenet64(data_root):
    trX = np.load(os.path.join(data_root, 'imagenet64-train.npy'), mmap_mode='r')
    tr_va_split_indices = np.random.permutation(trX.shape[0])
    train = trX[tr_va_split_indices[:-5000]]
    valid = trX[tr_va_split_indices[-5000:]]
    test = np.load(os.path.join(data_root, 'imagenet64-valid.npy'), mmap_mode='r')
    return train, valid, test


def ffhq1024(data_root):
    return os.path.join(data_root, 'ffhq1024/train'), os.path.join(data_root, 'ffhq1024/valid'), os.path.join(data_root, 'ffhq1024/valid')


def ffhq256(data_root):
    trX = np.load(os.path.join(data_root, 'ffhq-256.npy'), mmap_mode='r')
    tr_va_split_indices = np.random.permutation(trX.shape[0])
    train = trX[tr_va_split_indices[:-7000]]
    valid = trX[tr_va_split_indices[-7000:]]
    return train, valid, valid

def stl10(data_root):

    dataset = STL10("./data_stl", split="unlabeled", transform=transforms.Compose([
                            transforms.Resize(64),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), download=True)

    return dataset, None, None

class HuggingFaceLSUNChurch(Dataset):
    def __init__(self, data_root, split='train', image_size=256):
        self.ds = load_dataset("tglcourse/lsun_church_train", split=split)
        self.transform = transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        example = self.ds[idx]
        image = example["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = self.transform(image) * 255.0
        return [image]

def lsun_church(data_root):
    train_ds = HuggingFaceLSUNChurch(data_root, split='train', image_size=256)
    return train_ds, None, None


def cifar10(data_root, one_hot=True):
    tr_data = [unpickle_cifar10(os.path.join(data_root, 'cifar-10-batches-py/', 'data_batch_%d' % i)) for i in range(1, 6)]
    trX = np.vstack([data['data'] for data in tr_data])
    trY = np.asarray(flatten([data['labels'] for data in tr_data]))
    te_data = unpickle_cifar10(os.path.join(data_root, 'cifar-10-batches-py/', 'test_batch'))
    teX = np.asarray(te_data['data'])
    teY = np.asarray(te_data['labels'])
    trX = trX.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    teX = teX.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    if one_hot:
        trY = np.eye(10, dtype=np.float32)[trY]
        teY = np.eye(10, dtype=np.float32)[teY]
    else:
        trY = np.reshape(trY, [-1, 1])
        teY = np.reshape(teY, [-1, 1])
    return (trX, trY), (teX, teY)
