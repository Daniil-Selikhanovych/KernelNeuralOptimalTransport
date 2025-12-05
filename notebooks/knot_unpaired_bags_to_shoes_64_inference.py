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

from src.tools import freeze, load_dataset, get_Z_pushed_loader_stats, get_Z_pushed_loader_stats_resize
from src.fid_score import calculate_frechet_distance
from src.cunet import CUNet

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
path_to_dtd = os.path.join(path_to_data, "dtd/images")

# DATASET2, DATASET2_PATH = 'handbag', '../../data/handbag_128.hdf5'
# DATASET1, DATASET1_PATH = 'dtd', path_to_dtd
# DATASET2, DATASET2_PATH = 'shoes', path_to_shoes

DATASET1, DATASET1_PATH = 'handbag', path_to_bags
DATASET2, DATASET2_PATH = 'shoes', path_to_shoes

# DATASET1, DATASET1_PATH = 'outdoor', '../../data/outdoor_128.hdf5'
# DATASET2, DATASET2_PATH = 'church', '../../data/church_128.hdf5'

# DATASET1, DATASET1_PATH = 'celeba_female', '../../data/img_align_celeba'
# DATASET2, DATASET2_PATH = 'aligned_anime_faces', '../../data/aligned_anime_faces'

IMG_SIZE = 128
resize_shape = 64
COST = 'energy'

ZC, Z_STD = 128, 1.
    
assert torch.cuda.is_available()
torch.cuda.set_device(f'cuda:{DEVICE_ID}')

assert torch.cuda.is_available()
torch.cuda.set_device(f'cuda:{DEVICE_ID}')

AUGMENTED_DATASETS = ['dtd']
FID_EPOCHS = 50 if DATASET1 in AUGMENTED_DATASETS else 1
print(f"FID_EPOCHS = {FID_EPOCHS}")

# filename = f'{path_to_data}/{DATASET2}_{resize_shape}_from_shoes_64_test.json'
filename = f'{path_to_data}/{DATASET2}_{resize_shape}_test.json'

print(f"target dataset = {filename}")
with open(filename, 'r') as fp:
    data_stats = json.load(fp)
    mu_data, sigma_data = data_stats['mu'], data_stats['sigma']
del data_stats

_, X_test_sampler = load_dataset(DATASET1, DATASET1_PATH, img_size=IMG_SIZE, batch_size=256)
# _, Y_test_sampler = load_dataset(DATASET2, DATASET2_PATH, img_size=IMG_SIZE, batch_size=256)
    
T = CUNet(3, 3, ZC, base_factor=48)
pass

path_to_knot_ckpts = "/trinity/home/daniil.selikhanovych/my_thesis/knot_checkpoints"
folder = os.path.join(path_to_knot_ckpts, COST, '{}_{}_{}'.format(DATASET1, DATASET2, IMG_SIZE))
model = 'T.pt'
path = os.path.join(folder, model)

T.load_state_dict(torch.load(path))
T.cuda(); freeze(T)
torch.cuda.empty_cache()

torch.manual_seed(0xBADBEEF)
np.random.seed(0xBADBEEF)

num_calculation_fid = 10

fid_values = []

for _ in range(num_calculation_fid):
    mu, sigma = get_Z_pushed_loader_stats_resize(
        T, X_test_sampler.loader, ZC=ZC, Z_STD=Z_STD,
        n_epochs=FID_EPOCHS, verbose=True, use_downloaded_weights=True,
        resize_shape=resize_shape
    )
    fid = calculate_frechet_distance(mu_data, sigma_data, mu, sigma)
    print(f"FID = {fid}")
    fid_values.append(fid)
fid_values = np.array(fid_values)
fid_mean = np.mean(fid_values)
fid_std = np.std(fid_values)
print("--------")
print(f"Mean FID = {fid_mean}")
print(f"Std FID = {fid_std}")
