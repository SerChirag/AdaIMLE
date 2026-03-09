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

from helpers.utils import ZippedDataset, get_world_size
from models import parse_layer_string
from torchvision.datasets import CIFAR10, STL10


def set_up_data(H):
    
    blocks = parse_layer_string(H.dec_blocks)
    H.block_res = [s[0] for s in blocks]
    H.res = sorted(set([s[0] for s in blocks if s[0] <= H.max_hierarchy]))

    shift_loss = -127.5
    scale_loss = 1. / 127.5
    if H.dataset == 'imagenet32':
        (trX, trY) = imagenet32(H.data_root)
        vaX = None
        teX = None
        H.image_size = 32
        H.image_channels = 3
        shift = -116.2373
        scale = 1. / 69.37404
    elif H.dataset == 'imagenet32_200':
        (trX, trY) = imagenet32_200(H.data_root)
        vaX = None
        teX = None
        H.image_size = 32
        H.image_channels = 3
        shift = -116.2373
        scale = 1. / 69.37404   
    elif H.dataset == 'imagenet32_100':
        (trX, trY) = imagenet32_100(H.data_root)
        vaX = None
        teX = None
        H.image_size = 32
        H.image_channels = 3
        shift = -116.2373
        scale = 1. / 69.37404
    elif H.dataset in ['fewshot', 'fewshot512', 'fewshot64']:
        trX, vaX, teX = few_shot_image_folder(H.data_root, H.image_size)
        H.image_channels = 3
        shift = -116.2373
        scale = 1. / 69.37404
    elif H.dataset == 'imagenet64':
        (trX, trY) = imagenet64(H.data_root)
        vaX = None
        teX = None
        H.image_size = 64
        H.image_channels = 3
        shift = -115.92961967
        scale = 1. / 69.37404
    elif H.dataset == 'tinyimagenet64':
        (trX, trY) = tinyimagenet64(H.data_root)
        vaX = None
        teX = None
        H.image_size = 64
        H.image_channels = 3
        shift = -115.92961967
        scale = 1. / 69.37404
    elif H.dataset == 'smallimagenet64':
        (trX, trY) = smallimagenet64(H.data_root)
        vaX = None
        teX = None
        H.image_size = 64
        H.image_channels = 3
        shift = -115.92961967
        scale = 1. / 69.37404
    elif H.dataset == 'ffhq_256':
        trX, vaX, teX = ffhq256(H.data_root)
        H.image_size = 256
        H.image_channels = 3
        shift = -112.8666757481
        scale = 1. / 69.84780273
    elif H.dataset == 'ffhq_1024':
        trX, vaX, teX = ffhq1024(H.data_root)
        H.image_size = 1024
        H.image_channels = 3
        shift = -0.4387
        scale = 1.0 / 0.2743
        shift_loss = -0.5
        scale_loss = 2.0
    elif H.dataset == 'cifar10':
        (trX, trY), (vaX, varY), (teX, teY) = cifar10(H.data_root, one_hot=False)
        H.image_size = 32
        H.image_channels = 3
        shift = -120.63838
        scale = 1. / 64.16736
    elif H.dataset == "stl10":
        trX, vaX, teX = stl10(H.data_root)
        H.image_size = 64
        H.image_channels = 3
        shift = -0.5    
        scale = 1.0 / 0.5
    elif H.dataset == 'lsun':
        trX, vaX, teX = lsun_church(H.data_root)   # helper above
        H.image_size     = 256
        H.image_channels = 3
        shift = -0.5               
        scale = 1. / 0.5
    else:
        raise ValueError('unknown dataset: ', H.dataset)

    do_low_bit = H.dataset in ['ffhq_256']

    if H.test_eval:
        print('DOING TEST')
        eval_dataset = teX
    else:
        eval_dataset = None

    device = torch.device("cuda", torch.cuda.current_device())

    shift = torch.tensor([shift], device=device).view(1, 1, 1, 1)
    scale = torch.tensor([scale], device=device).view(1, 1, 1, 1)
    shift_loss = torch.tensor([shift_loss], device=device).view(1, 1, 1, 1)
    scale_loss = torch.tensor([scale_loss], device=device).view(1, 1, 1, 1)

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
    
    elif H.dataset == 'imagenet32' or H.dataset == 'imagenet32_200' or H.dataset == 'imagenet32_100':
        train_data = TensorDataset(torch.as_tensor(trX).permute(0, 2, 3, 1), torch.as_tensor(trY))
        valid_data = None
        train_len = len(train_data)
        untranspose = False
    
    elif H.dataset == 'tinyimagenet64':
        train_data = TensorDataset(torch.as_tensor(trX).permute(0, 2, 3, 1), torch.as_tensor(trY))
        valid_data = None
        train_len = len(train_data)
        untranspose = False
    
    elif H.dataset == 'smallimagenet64':
        train_data = TensorDataset(torch.as_tensor(trX).permute(0, 2, 3, 1), torch.as_tensor(trY))
        valid_data = None
        train_len = len(train_data)
        untranspose = False
    
    elif H.dataset == 'imagenet64':
        train_data = TensorDataset(torch.as_tensor(trX).permute(0, 2, 3, 1), torch.as_tensor(trY))
        valid_data = None
        train_len = len(train_data)
        untranspose = False

    elif H.dataset not in ['fewshot', 'fewshot512']:
        train_data = TensorDataset(torch.as_tensor(trX), torch.as_tensor(trY))
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
    H.total_iters = H.num_epochs * np.ceil(train_len // H.global_batch_size)


    def preprocess_func(x):
        nonlocal shift
        nonlocal scale
        nonlocal shift_loss
        nonlocal scale_loss
        nonlocal do_low_bit
        nonlocal untranspose
        'takes in a data example and returns the preprocessed input'
        'as well as the input processed for the loss'
        if untranspose:
            x = x.permute(0, 2, 3, 1)
        inp = x.to(device=device, non_blocking=True).float()
        inp.mul_(1./127.5).add_(-1)
        # out = inp.clone()
        # inp.add_(shift).mul_(scale)
        # if do_low_bit:
        #     5 bits of precision
        #     out.mul_(1. / 8.).floor_().mul_(8.)
        # out.add_(shift_loss).mul_(scale_loss)
        return inp

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

    sort_indices = np.argsort(labels)
    images = images[sort_indices]
    labels = labels[sort_indices]

    return (images, labels)

def imagenet32_200(data_root):

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

    sort_indices = np.argsort(labels)
    images = images[sort_indices]
    labels = labels[sort_indices]

      # Select first 200 classes
    chosen_classes = np.arange(200)
    mask = np.isin(labels, chosen_classes)
    images = images[mask]
    labels = labels[mask]

    return (images, labels)

def imagenet32_100(data_root):

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

    sort_indices = np.argsort(labels)
    images = images[sort_indices]
    labels = labels[sort_indices]

    # Select first 100 classes
    chosen_classes = np.arange(100)
    mask = np.isin(labels, chosen_classes)
    images = images[mask]
    labels = labels[mask]

    return (images, labels)


def imagenet64(data_root):

    files = sorted([f for f in os.listdir(data_root) if f.endswith(".npz")])

    images, labels = [], []

    for f in files:
        batch = np.load(os.path.join(data_root, f))
        X = batch["data"]        # shape (N, 3072)
        Y = batch["labels"]      # shape (N,)
        
        X = X.reshape(-1, 3, 64, 64)
        images.append(X)
        labels.append(Y)

    images = np.concatenate(images)
    labels = np.concatenate(labels) - 1

    sort_indices = np.argsort(labels)
    images = images[sort_indices]
    labels = labels[sort_indices]


    return (images, labels)


def tinyimagenet64(data_root):

    batch = np.load(data_root, mmap_mode="r")  # mmap = no RAM spike

    images = batch["data"]    # (N, 3, 64, 64), uint8
    labels = batch["labels"]  # (N,), int64

    return images, labels

def smallimagenet64(data_root):

    files = sorted([f for f in os.listdir(data_root) if f.endswith(".npz")])

    images, labels = [], []

    for f in files:
        batch = np.load(os.path.join(data_root, f))
        X = batch["data"]        # shape (N, 3072)
        Y = batch["labels"]      # shape (N,)

        X = X.reshape(-1, 3, 64, 64)
        images.append(X)
        labels.append(Y)


    images = np.concatenate(images)
    labels = np.concatenate(labels) - 1

    # sort by labels
    sort_indices = np.argsort(labels)
    images = images[sort_indices]
    labels = labels[sort_indices]

  # Select first 100 classes
    chosen_classes = np.arange(50)
    mask = np.isin(labels, chosen_classes)
    images = images[mask]
    labels = labels[mask]

    return (images, labels)


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
    sort_indices = np.argsort(trY)
    trX = trX[sort_indices]
    trY = trY[sort_indices]
    if one_hot:
        trY = np.eye(10, dtype=np.float32)[trY]
    else:
        trY = np.reshape(trY, [-1, 1])
    return (trX, trY), (None, None), (None, None)
