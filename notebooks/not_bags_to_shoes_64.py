import os, sys
sys.path.append("..")

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline 

import numpy as np
import torch
import torch.nn as nn
import torchvision
import gc

from src.tools import freeze, load_dataset, get_pushed_loader_stats
from src.fid_score import calculate_frechet_distance
from src.unet import UNet

import json

from tqdm import tqdm
# from IPython.display import clear_output
from collections import OrderedDict

# This needed to use dataloaders for some datasets
# from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
# PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

DEVICE_ID = 0

path_to_data = "/trinity/home/daniil.selikhanovych/my_thesis/datasets"
path_to_shoes = os.path.join(path_to_data, "shoes_128.hdf5")
path_to_bags = os.path.join(path_to_data, "handbag_128.hdf5")
path_to_shoes_64 = os.path.join(path_to_data, "shoes_64.hdf5")
path_to_bags_64 = os.path.join(path_to_data, "handbag_64.hdf5")

# DATASET2, DATASET2_PATH = 'handbag', '../../data/handbag_128.hdf5'
# DATASET1, DATASET1_PATH = 'dtd', path_to_dtd
# DATASET2, DATASET2_PATH = 'shoes', path_to_shoes

DATASET1, DATASET1_PATH = 'handbag', path_to_bags_64
DATASET2, DATASET2_PATH = 'shoes', path_to_shoes_64

# DATASET1, DATASET1_PATH = 'outdoor', '../../data/outdoor_128.hdf5'
# DATASET2, DATASET2_PATH = 'church', '../../data/church_128.hdf5'
resize_shape = 64
# DATASET1, DATASET1_PATH = 'celeba_female', '../../data/img_align_celeba'
# DATASET2, DATASET2_PATH = 'aligned_anime_faces', '../../data/aligned_anime_faces'

COST = 'mse'

ZC, Z_STD = 0, 0.
    
assert torch.cuda.is_available()
torch.cuda.set_device(f'cuda:{DEVICE_ID}')

# filename = f'{path_to_data}/{DATASET2}_{resize_shape}_from_shoes_64_test.json'
filename = f'{path_to_data}/{DATASET2}_{resize_shape}_test.json'

print(f"target dataset = {filename}")
with open(filename, 'r') as fp:
    data_stats = json.load(fp)
    mu_data, sigma_data = data_stats['mu'], data_stats['sigma']
del data_stats

_, X_test_sampler = load_dataset(DATASET1, DATASET1_PATH, img_size=resize_shape, batch_size=256)
# _, Y_test_sampler = load_dataset(DATASET2, DATASET2_PATH, img_size=IMG_SIZE, batch_size=256)
    
T = UNet(3+ZC, 3, base_factor=48)
pass

path_to_not_ckpts = "/trinity/home/daniil.selikhanovych/my_thesis/not_checkpoints"
folder = os.path.join(path_to_not_ckpts, COST, '{}_{}_{}'.format(DATASET1, DATASET2, resize_shape))
model = 'T.pt'
path = os.path.join(folder, model)
print(f"loading {path}")

T.load_state_dict(torch.load(path))
T.cuda(); freeze(T)
torch.cuda.empty_cache()

torch.manual_seed(0xBADBEEF)
np.random.seed(0xBADBEEF)

batch_size = 64

device = 'cuda:0'


freeze(T)
    
size = len(X_test_sampler.dataset)
pred_arr = []

with torch.no_grad():
    for step, (X, _) in tqdm(enumerate(X_test_sampler)):
        for i in range(0, len(X), batch_size):
            start, end = i, min(i + batch_size, len(X))
            batch = T(
                X[start:end].type(torch.FloatTensor).to(device),
            ).add(1).mul(.5)
            pred_arr.append(batch.cpu().permute((0, 2, 3, 1)).data.numpy())

pred_arr = np.vstack(pred_arr)
print(f"pred_arr.shape = {pred_arr.shape}")
