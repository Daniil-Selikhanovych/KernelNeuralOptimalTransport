import os
import glob
import h5py
import torch
import numpy as np
import shutil
from tqdm import tqdm
from PIL import Image
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader, Dataset
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, Lambda, Pad, CenterCrop, RandomResizedCrop
from torchvision.datasets import ImageFolder

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

def get_subset_image_paths(subset):
    # Get the original dataset from the subset
    original_dataset = subset.dataset
    
    # Get the indices used in the subset
    subset_indices = subset.indices
    
    # Extract paths using the original dataset's samples
    paths = []
    for idx in subset_indices:
        # ImageFolder stores paths in dataset.samples
        path, label = original_dataset.samples[idx]
        paths.append(path)
    
    return paths

path_shoes = "/trinity/home/daniil.selikhanovych/my_thesis/datasets/outdoor_128.hdf5"
img_size = 128

dataset = h5py_to_dataset(path_shoes, img_size)

idx = list(range(len(dataset)))

test_ratio=0.1
test_size = int(len(idx) * test_ratio)

train_idx, test_idx = idx[:-test_size], idx[-test_size:]

train_set, test_set = Subset(dataset, train_idx), Subset(dataset, test_idx)

batch_size = 1

train_dataloader = DataLoader(train_set, shuffle=True, num_workers=8, batch_size=batch_size, drop_last=False)
test_dataloader = DataLoader(test_set, shuffle=True, num_workers=8, batch_size=batch_size, drop_last=False)

path_to_save_splits = "/trinity/home/daniil.selikhanovych/OptimalTransportModeling/data/outdoor"
os.makedirs(path_to_save_splits, exist_ok=True)

path_to_save_train_shoes = os.path.join(path_to_save_splits, f"train_outdoor_{img_size}", "outdoor")
path_to_save_test_shoes = os.path.join(path_to_save_splits, f"test_outdoor_{img_size}", "outdoor")

os.makedirs(path_to_save_train_shoes, exist_ok=True)
os.makedirs(path_to_save_test_shoes, exist_ok=True)

index = 0
for (X, _) in tqdm(train_dataloader):
    image_name = f"train_{index}.png"
    path_to_save_img = os.path.join(path_to_save_train_shoes, image_name)
    X_numpy = (((X + 1) * 0.5) * 255).permute((0, 2, 3, 1))[0].cpu().numpy().astype(np.uint8)
    # print(f"X_numpy.shape = {X_numpy.shape}")
    im = Image.fromarray(X_numpy)
    im.save(path_to_save_img)
    index += 1

index = 0
for (X, _) in tqdm(test_dataloader):
    image_name = f"test_{index}.png"
    path_to_save_img = os.path.join(path_to_save_test_shoes, image_name)
    X_numpy = (((X + 1) * 0.5) * 255).permute((0, 2, 3, 1))[0].cpu().numpy().astype(np.uint8)
    # print(f"X_numpy.shape = {X_numpy.shape}")
    im = Image.fromarray(X_numpy)
    im.save(path_to_save_img)
    index += 1