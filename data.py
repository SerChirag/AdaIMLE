import numpy as np
import pickle
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset

from helpers.utils import get_world_size
from models import parse_layer_string
from torchvision.datasets import CIFAR10, STL10
from helpers.autoencoder import load_autoencoder, encode_images_to_latents


def set_up_data(H):
    
    blocks = parse_layer_string(H.dec_blocks)
    H.block_res = [s[0] for s in blocks]
    H.res = sorted(set([s[0] for s in blocks if s[0] <= H.max_hierarchy]))
    H.latent_spatial_size = max(H.block_res)

    if H.dataset == 'imagenet32':
        trX, vaX, teX = imagenet32(H.data_root)
        H.image_size = 32
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
        (trX, _), (teX, _) = cifar10(H.data_root, one_hot=False)
        H.image_size = 32
        H.image_channels = 3
    elif H.dataset == "stl10":
        trX, vaX, teX = stl10(H.data_root)
        H.image_size = 64
        H.image_channels = 3
    elif H.dataset == 'lsun':
        trX, vaX, teX = lsun_church(H.data_root)   # helper above
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

    # if H.dataset == 'ffhq_1024':
    #     train_data = ImageFolder(trX, transforms.ToTensor())
    #     valid_data = ImageFolder(eval_dataset, transforms.ToTensor())
    #     untranspose = True
    train_len = None
    if H.dataset == 'stl10':
        train_data = trX
        for data_train in DataLoader(train_data, batch_size=len(train_data)):
            ds = torch.tensor((data_train[0] + 1)/2 * 255, dtype=torch.uint8)
            train_data = TensorDataset(ds.permute(0, 2, 3, 1))
            break
        valid_data = train_data
        untranspose = False
        train_len = len(train_data)
    
    elif H.dataset == 'lsun':
        train_data = trX
        valid_data = trX
        train_len = train_data.ds.num_rows  
        untranspose = True
    
    elif H.dataset == 'imagenet32':
        train_data = TensorDataset(torch.as_tensor(trX).permute(0, 2, 3, 1))
        valid_data = None
        train_len = len(train_data)
        untranspose = False

    elif H.dataset not in ['fewshot', 'fewshot512', 'fewshot64']:
        train_data = TensorDataset(torch.as_tensor(trX))
        valid_data = None
        untranspose = False
        train_len = len(train_data)

    else:
        train_data = trX
        for data_train in DataLoader(train_data, batch_size=len(train_data)):
            ds = torch.tensor(data_train[0] * 255, dtype=torch.uint8)
            train_data = TensorDataset(ds.permute(0, 2, 3, 1))
            break
        valid_data = train_data
        untranspose = False
        train_len = len(train_data)
    
        
    H.global_batch_size = H.n_batch * get_world_size()
    H.total_iters = H.num_epochs * ((train_len + H.global_batch_size - 1) // H.global_batch_size)

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
        return inp, target

    return H, train_data, valid_data, preprocess_func


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

    return images, None, None


def imagenet64(data_root):
    trX = np.load(os.path.join(data_root, 'imagenet64-train.npy'), mmap_mode='r')
    tr_va_split_indices = np.random.permutation(trX.shape[0])
    train = trX[tr_va_split_indices[:-5000]]
    valid = trX[tr_va_split_indices[-5000:]]
    test = np.load(os.path.join(data_root, 'imagenet64-valid.npy'), mmap_mode='r')  # this is test.
    return train, valid, test


def ffhq1024(data_root):
    # we did not significantly tune hyperparameters on ffhq-1024, and so simply evaluate on the test set
    return os.path.join(data_root, 'ffhq1024/train'), os.path.join(data_root, 'ffhq1024/valid'), os.path.join(data_root, 'ffhq1024/valid')


def ffhq256(data_root):
    trX = np.load(os.path.join(data_root, 'ffhq-256.npy'), mmap_mode='r')
    tr_va_split_indices = np.random.permutation(trX.shape[0])
    train = trX[tr_va_split_indices[:-7000]]
    valid = trX[tr_va_split_indices[-7000:]]
    # we did not significantly tune hyperparameters on ffhq-256, and so simply evaluate on the test set
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
    return train_ds, None, None  # No validation or test set in this case


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
