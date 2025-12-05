import os, sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import gc

from src.tools import freeze
from src.tools import load_dataset
from src.cunet import CUNet

import json

from tqdm import tqdm

DEVICE_ID = 0

path_to_data = "/trinity/home/daniil.selikhanovych/my_thesis/datasets"
path_to_shoes = os.path.join(path_to_data, "shoes_128.hdf5")
path_to_dtd = os.path.join(path_to_data, "dtd/images")
path_to_handbags = os.path.join(path_to_data, "handbag_128.hdf5")

# DATASET2, DATASET2_PATH = 'handbag', '../../data/handbag_128.hdf5'
DATASET1, DATASET1_PATH = 'handbag', path_to_handbags
DATASET2, DATASET2_PATH = 'shoes', path_to_shoes

IMG_SIZE = 128
COST = 'energy' #'weak_mse'

ZC, Z_STD = 128, 1.
    
assert torch.cuda.is_available()

_, X_test_sampler = load_dataset(DATASET1, DATASET1_PATH, img_size=IMG_SIZE)
# _, Y_test_sampler = load_dataset(DATASET2, DATASET2_PATH, img_size=IMG_SIZE) <-- not needed in this notebook
    
T = CUNet(3, 3, ZC, base_factor=48)

device = 'cuda:0'

path_to_knot_ckpts = "/trinity/home/daniil.selikhanovych/my_thesis/knot_checkpoints"
folder = os.path.join(path_to_knot_ckpts, COST, '{}_{}_{}'.format(DATASET1, DATASET2, IMG_SIZE))
model = 'T.pt'
path = os.path.join(folder, model)

T.load_state_dict(torch.load(path))
T.to(device); freeze(T)

import h5py
import numpy as np
import gc

import torch
from torch.utils.data import TensorDataset
import torch.nn.functional as F

from torchvision.transforms import Compose, Resize, Normalize, ToTensor, \
    RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, DataLoader
import torch.nn as nn

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
    elif name in ['dtd']:
        transform = Compose(
            [Resize(300), RandomResizedCrop((img_size,img_size), scale=(128./300, 1.), ratio=(1., 1.)),
             RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5),
             ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
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


class Sampler:
    def __init__(
            self, device='cuda',
    ):
        self.device = device

    def sample(self, size=5):
        pass


class LoaderSampler(Sampler):
    def __init__(self, loader, device='cuda'):
        super(LoaderSampler, self).__init__(device)
        self.loader = loader
        self.it = iter(self.loader)

    def sample(self, size=5):
        assert size <= self.loader.batch_size
        try:
            batch, _ = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample(size)
        if len(batch) < size:
            return self.sample(size)

        return batch[:size].to(self.device)

def tensor2img(tensor):
    return tensor.to('cpu').permute(0, 2, 3, 1).mul(0.5).add(0.5).numpy().clip(0, 1)    

input_shape = (3, IMG_SIZE, IMG_SIZE)

torch.manual_seed(0xBADBEEF)
np.random.seed(0xBADBEEF)

print(f"loading a")
train_loader_a, test_loader_a = load_dataset(DATASET1, DATASET1_PATH,
                                             img_size=IMG_SIZE, batch_size=32)
print(f"loading b")
train_loader_b, test_loader_b = load_dataset(DATASET2, DATASET2_PATH,
                                             img_size=IMG_SIZE, batch_size=32)

n_batches = min(len(train_loader_a), len(train_loader_b))

X_sampler = LoaderSampler(train_loader_a, device=device)
X_test_sampler = LoaderSampler(test_loader_a, device=device)
Y_sampler = LoaderSampler(train_loader_b, device=device)
Y_test_sampler = LoaderSampler(test_loader_b, device=device)

torch.manual_seed(0xBADBEEF)
np.random.seed(0xBADBEEF)
X_fixed = X_sampler.sample(12)
Y_fixed = Y_sampler.sample(12)

X_test_fixed = X_test_sampler.sample(12)
Y_test_fixed = Y_test_sampler.sample(12)

num_examples = 4
Y_test_fakes = []

X_test_fixed_2 = X_test_fixed[:4,None].repeat(1,4,1,1,1)

Z = torch.randn(4, 4, ZC, 1, 1, device='cuda') * Z_STD
T_XZ = T(
        X_test_fixed_2.flatten(start_dim=0, end_dim=1), Z.flatten(start_dim=0, end_dim=1)
    ).permute(1,2,3,0).reshape(3, IMG_SIZE, IMG_SIZE, -1, 4).permute(3,4,0,1,2)

fig, axes = plt.subplots(5, 4, figsize=(11, 13), dpi=100)
X_tensor = tensor2img(X_test_fixed)

for j in range(4):
    axes[0][j].imshow(X_tensor[j])
    axes[0][j].get_xaxis().set_visible(False)
    axes[0][j].set_yticks([])
    #axes[0][j].get_yaxis().set_visible(False)

for i in range(4):
    for j in range(4):
        cur_img = T_XZ[j][i].permute(1, 2, 0).add(1).mul(0.5).cpu().numpy().clip(0,1).reshape(128, 128, 3)

        axes[i + 1][j].imshow(cur_img)
        axes[i + 1][j].get_xaxis().set_visible(False)
        axes[i + 1][j].set_yticks([])
        #axes[i + 1][j].get_yaxis().set_visible(False)
        
axes[0, 0].set_ylabel(r'$x\sim\mathbb{P}$', fontsize=30)
for j in range(4):
    title = '\widehat{T}(x,z_' + str(j+1) + ')'
    axes[j+1, 0].set_ylabel(r'${}$'.format(title), fontsize=30)
    
fig.tight_layout(pad=0.001)    
plt.savefig("knot_textures2shoes_test.png")