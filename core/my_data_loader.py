import os
from pathlib import Path
from itertools import chain
import random

from PIL import Image
import numpy as np
import h5py
from munch import Munch

import torch
from torch.utils.data import TensorDataset
from torch.utils import data
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomResizedCrop, RandomHorizontalFlip, \
    RandomVerticalFlip
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, DataLoader
import torch.nn.functional as F
from torch.utils.data.sampler import WeightedRandomSampler


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


class ReferenceDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples, self.targets = self._make_dataset(root)
        self.transform = transform

    def _make_dataset(self, root):
        domains = os.listdir(root)
        fnames, fnames2, labels = [], [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames
            fnames2 += random.sample(cls_fnames, len(cls_fnames))
            labels += [idx] * len(cls_fnames)
        return list(zip(fnames, fnames2)), labels

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        label = self.targets[index]
        img = Image.open(fname).convert('RGB')
        img2 = Image.open(fname2).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return img, img2, label

    def __len__(self):
        return len(self.targets)


class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def __next__(self):
        x, y = self._fetch_inputs()
        if self.mode == 'train':
            x_ref, x_ref2, y_ref = self._fetch_refs()
            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(x_src=x, y_src=y, y_ref=y_ref,
                           x_ref=x_ref, x_ref2=x_ref2,
                           z_trg=z_trg, z_trg2=z_trg2)
        elif self.mode == 'val':
            x_ref, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, y_src=y,
                           x_ref=x_ref, y_ref=y_ref)
        elif self.mode == 'test':
            inputs = Munch(x=x, y=y)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def get_train_loader(root, which='source', img_size=256,
                     batch_size=8, num_workers=4):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)

    transform = Compose([Resize((img_size, img_size)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if which == 'source':
        dataset = ImageFolder(root, transform)
    elif which == 'reference':
        dataset = ReferenceDataset(root, transform)
    else:
        raise NotImplementedError

    sampler = _make_balanced_sampler(dataset.targets)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           sampler=sampler,
                           num_workers=num_workers,
                           pin_memory=True)


def get_test_loader(root, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4):
    print('Preparing DataLoader for the generation phase...')
    transform = Compose([Resize((img_size, img_size)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = ImageFolder(root, transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True)


def h5py_to_dataset(path, img_size=64):
    with h5py.File(path, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = list(f[a_group_key])
    with torch.no_grad():
        dataset = 2 * (torch.tensor(np.array(data), dtype=torch.float32) / 255.).permute(0, 3, 1, 2) - 1
        dataset = F.interpolate(dataset, img_size, mode='bilinear')

    return TensorDataset(dataset, torch.zeros(len(dataset)))


def load_dataset(name, path, img_size=64, batch_size=64, test_ratio=0.1):
    if name in ['shoes', 'handbag', 'outdoor', 'church']:
        dataset = h5py_to_dataset(path, img_size)
    elif name in ['celeba_female', 'celeba_male', 'aligned_anime_faces']:
        transform = Compose([Resize((img_size, img_size)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = ImageFolder(path, transform=transform)
    else:
        raise Exception('Unknown dataset')

    if name in ['celeba_female', 'celeba_male']:
        with open('datasets/list_attr_celeba.txt', 'r') as f:
            lines = f.readlines()[2:]
        if name == 'celeba_female':
            idx = [i for i in list(range(len(lines))) if lines[i].replace('  ', ' ').split(' ')[21] == '-1']
        else:
            idx = [i for i in list(range(len(lines))) if lines[i].replace('  ', ' ').split(' ')[21] != '-1']
    else:
        idx = list(range(len(dataset)))

    test_size = int(len(idx) * test_ratio)
    train_idx, test_idx = idx[:-test_size], idx[-test_size:]
    train_set, test_set = Subset(dataset, train_idx), Subset(dataset, test_idx)

    train_dataloader = DataLoader(train_set, shuffle=True, num_workers=8, batch_size=batch_size)
    test_dataloader = DataLoader(test_set, shuffle=True, num_workers=8, batch_size=batch_size)
    return train_dataloader, test_dataloader


def load_dataset_h5py_no_transform(path, batch_size=64, test_ratio=0.1):
    with h5py.File(path, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]
        # Get the data
        data = list(f[a_group_key])
    with torch.no_grad():
        dataset = torch.tensor(np.array(data), dtype=torch.float32).permute(0, 3, 1, 2)

    dataset = TensorDataset(dataset, torch.zeros(len(dataset)))
    idx = list(range(len(dataset)))

    test_size = int(len(idx) * test_ratio)
    train_idx, test_idx = idx[:-test_size], idx[-test_size:]
    train_set, test_set = Subset(dataset, train_idx), Subset(dataset, test_idx)

    train_dataloader = DataLoader(train_set, shuffle=True, num_workers=8, batch_size=batch_size)
    test_dataloader = DataLoader(test_set, shuffle=True, num_workers=8, batch_size=batch_size)
    return train_dataloader, test_dataloader


def load_dataset_celeba_no_transform(name, path, batch_size=64, test_ratio=0.1):
    transform = Compose([ToTensor()])
    dataset = ImageFolder(path, transform=transform)

    if name in ['celeba_female', 'celeba_male']:
        with open('datasets/list_attr_celeba.txt', 'r') as f:
            lines = f.readlines()[2:]
        if name == 'celeba_female':
            idx = [i for i in list(range(len(lines))) if lines[i].replace('  ', ' ').split(' ')[21] == '-1']
        else:
            idx = [i for i in list(range(len(lines))) if lines[i].replace('  ', ' ').split(' ')[21] != '-1']
    else:
        idx = list(range(len(dataset)))

    test_size = int(len(idx) * test_ratio)
    train_idx, test_idx = idx[:-test_size], idx[-test_size:]
    train_set, test_set = Subset(dataset, train_idx), Subset(dataset, test_idx)

    train_dataloader = DataLoader(train_set, shuffle=True, num_workers=8, batch_size=batch_size)
    test_dataloader = DataLoader(test_set, shuffle=True, num_workers=8, batch_size=batch_size)
    return train_dataloader, test_dataloader


def load_dataset_dtd_no_transform(path, img_size, batch_size=64, test_ratio=0.1):
    transform = Compose([Resize(300),
                         RandomResizedCrop((img_size, img_size), scale=(128./300, 1.), ratio=(1., 1.)),
                         RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5), ToTensor()])
    dataset = ImageFolder(path, transform=transform)
    idx = list(range(len(dataset)))

    test_size = int(len(idx) * test_ratio)
    train_idx, test_idx = idx[:-test_size], idx[-test_size:]
    train_set, test_set = Subset(dataset, train_idx), Subset(dataset, test_idx)

    train_dataloader = DataLoader(train_set, shuffle=True, num_workers=8, batch_size=batch_size)
    test_dataloader = DataLoader(test_set, shuffle=True, num_workers=8, batch_size=batch_size)
    return train_dataloader, test_dataloader
