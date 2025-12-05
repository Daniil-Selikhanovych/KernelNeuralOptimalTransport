import os, sys
sys.path.append("..")

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
# %matplotlib inline 

import numpy as np
import torch
import torch.nn as nn
import torchvision
import gc

from src.tools import freeze, load_dataset, get_Z_pushed_loader_stats
from src.fid_score import calculate_frechet_distance
from src.cunet import CUNet
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
path_to_dtd = os.path.join(path_to_data, "dtd/images")

# DATASET2, DATASET2_PATH = 'handbag', '../../data/handbag_128.hdf5'
# DATASET1, DATASET1_PATH = 'dtd', path_to_dtd
DATASET1, DATASET1_PATH = 'handbag', path_to_bags
DATASET2, DATASET2_PATH = 'shoes', path_to_shoes

# DATASET1, DATASET1_PATH = 'outdoor', '../../data/outdoor_128.hdf5'
# DATASET2, DATASET2_PATH = 'church', '../../data/church_128.hdf5'

# DATASET1, DATASET1_PATH = 'celeba_female', '../../data/img_align_celeba'
# DATASET2, DATASET2_PATH = 'aligned_anime_faces', '../../data/aligned_anime_faces'

IMG_SIZE = 128
COST = 'weak_mse'

ZC, Z_STD = 128, 1.
    
assert torch.cuda.is_available()
torch.cuda.set_device(f'cuda:{DEVICE_ID}')

# assert torch.cuda.is_available()
# torch.cuda.set_device(f'cuda:{DEVICE_ID}')

# AUGMENTED_DATASETS = ['dtd']
# FID_EPOCHS = 50 if DATASET1 in AUGMENTED_DATASETS else 1

# filename = f'{path_to_data}/{DATASET2}_{IMG_SIZE}_test.json'
# with open(filename, 'r') as fp:
#     data_stats = json.load(fp)
#     mu_data, sigma_data = data_stats['mu'], data_stats['sigma']
# del data_stats

# _, X_test_sampler = load_dataset(DATASET1, DATASET1_PATH, img_size=IMG_SIZE, batch_size=256)
# _, Y_test_sampler = load_dataset(DATASET2, DATASET2_PATH, img_size=IMG_SIZE, batch_size=256)
    
# T = CUNet(3, 3, ZC, base_factor=48)
COST = 'weak_mse'

if COST == 'weak_mse':
    ZC, Z_STD = 1, 0.1
else:
    ZC, Z_STD = 0, 0.
pass

T = UNet(3+ZC, 3, base_factor=48)

path_to_not_ckpts = "/trinity/home/daniil.selikhanovych/my_thesis/not_checkpoints"
folder = os.path.join(path_to_not_ckpts, COST, '{}_{}_{}'.format(DATASET1, DATASET2, IMG_SIZE))
model = 'T.pt'
path = os.path.join(folder, model)

T.load_state_dict(torch.load(path))
T.cuda(); freeze(T)
torch.cuda.empty_cache()

torch.manual_seed(0xBADBEEF)
np.random.seed(0xBADBEEF)

train_loader_a, test_loader_a = load_dataset(DATASET1, DATASET1_PATH,
                                             img_size=IMG_SIZE, batch_size=32)

# n_batches = min(len(train_loader_a))

device = 'cuda:0'

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

# X_sampler = LoaderSampler(train_loader_a, device=device)
# X_test_sampler = LoaderSampler(test_loader_a, device=device)
X_sampler = train_loader_a
X_test_sampler = test_loader_a

torch.manual_seed(0xBADBEEF)
np.random.seed(0xBADBEEF)
X_fixed = X_sampler.sample(12)

X_test_fixed = X_test_sampler.sample(12)

num_examples = 4
Y_test_fakes = []

X_test_fixed_2 = X_test_fixed[:4,None].repeat(1,4,1,1,1)

with torch.no_grad():
    Z = torch.randn(4, 4, ZC, IMG_SIZE, IMG_SIZE, device='cuda') * Z_STD
    XZ = torch.cat([X_test_fixed_2, Z], dim=2)
    T_XZ = T(
        XZ.flatten(start_dim=0, end_dim=1)
    ).permute(1,2,3,0).reshape(3, IMG_SIZE, IMG_SIZE, -1, 4).permute(3,4,0,1,2)

# Z = torch.randn(4, 4, ZC, 1, 1, device='cuda') * Z_STD
# T_XZ = T(
#         X_test_fixed_2.flatten(start_dim=0, end_dim=1), Z.flatten(start_dim=0, end_dim=1)
#     ).permute(1,2,3,0).reshape(3, IMG_SIZE, IMG_SIZE, -1, 4).permute(3,4,0,1,2)


def tensor2img(tensor):
    return tensor.to('cpu').permute(0, 2, 3, 1).mul(0.5).add(0.5).numpy().clip(0, 1)

fig, axes = plt.subplots(5, 4, figsize=(11, 13), dpi=100)
X_tensor = tensor2img(X_test_fixed)

img_index = 1

for j in range(4):
    axes[0][j].imshow(X_tensor[j])
    axes[0][j].get_xaxis().set_visible(False)
    axes[0][j].set_yticks([])
    #axes[0][j].get_yaxis().set_visible(False)

for i in range(4):
    for j in range(4):
        cur_img = T_XZ[j][i].permute(1, 2, 0).add(1).mul(0.5).cpu().numpy().clip(0,1).reshape(128, 128, 3)

        if i != 3 or j != 1:
            axes[i + 1][j].imshow((cur_img * 255).astype(np.uint8))
            axes[i + 1][j].get_xaxis().set_visible(False)
            axes[i + 1][j].set_yticks([])
        else:
            # input_img_raw = X_test_fixed.permute(0, 2, 3, 1)[img_index]
            
            # cur_img = T_XZ[img_index][1].permute(1, 2, 0)
            # cur_img = cur_img.permute(1, 2, 0).add(1).mul(0.5).cpu().numpy().clip(0,1).reshape(128, 128, 3)
            
            # input_img = X_tensor[img_index]
            # difference = input_img - cur_img
            
            axes[i + 1][j].imshow((cur_img * 255).astype(np.uint8))
            axes[i + 1][j].get_xaxis().set_visible(False)
            axes[i + 1][j].set_yticks([])
        #axes[i + 1][j].get_yaxis().set_visible(False)
        
axes[0, 0].set_ylabel(r'$x\sim\mathbb{P}$', fontsize=30)
for j in range(4):
    title = '\widehat{T}(x,z_' + str(j+1) + ')'
    axes[j+1, 0].set_ylabel(r'${}$'.format(title), fontsize=30)
    
fig.tight_layout(pad=0.001)    
plt.savefig("plot_not_bags2shoes_128.png")

img_index = 1

input_img = X_tensor[img_index]

fig, ax = plt.subplots(figsize=(10, 10))

# Display image
# ax.imshow((input_img * 255).astype(np.uint8))
ax.imshow((input_img * 255).astype(np.uint8), extent=[0, input_img.shape[1], 0, input_img.shape[0]])

# Create a rectangle border
border_width=2
border_color='black'
border = patches.Rectangle((0, 0), input_img.shape[1], input_img.shape[0],
                            linewidth=border_width, edgecolor=border_color, 
                            facecolor='none')
ax.add_patch(border)

ax.set_xlim(0, input_img.shape[1])
ax.set_ylim(0, input_img.shape[0])

ax.axis('off')
plt.tight_layout()
plt.savefig("input_not_bags2shoes_128.png")
plt.show()

# Image.fromarray((input_img * 255).astype(np.uint8)).save("input.png")

input_img_raw = X_test_fixed.permute(0, 2, 3, 1)[img_index]
print(f"min input = {input_img_raw.min()}, max = {input_img_raw.max()}")

print(f" T_XZ.shape = {T_XZ.shape}")
cur_img = T_XZ[img_index][1].permute(1, 2, 0)

print(f"cur_img.min = {cur_img.min()}, max = {cur_img.max()}")

cur_img = cur_img.add(1).mul(0.5).cpu().numpy().clip(0,1).reshape(128, 128, 3)

output_img = cur_img

print(f"output min = {np.min(output_img)}, max = {np.max(output_img)}")

fig, ax = plt.subplots(figsize=(10, 10))

# Display image
# ax.imshow((output_img * 255).astype(np.uint8))
ax.imshow((output_img * 255).astype(np.uint8), extent=[0, output_img.shape[1], 0, output_img.shape[0]])

# Create a rectangle border
border = patches.Rectangle((0, 0), output_img.shape[1], output_img.shape[0],
                            linewidth=border_width, edgecolor=border_color, 
                            facecolor='none')
ax.add_patch(border)

ax.set_xlim(0, output_img.shape[1])
ax.set_ylim(0, output_img.shape[0])
ax.axis('off')
plt.tight_layout()
plt.savefig("output_img_not_bags2shoes_128.png")
plt.show()

# Image.fromarray((output_img * 255).astype(np.uint8)).save("output_img.png")

difference = np.abs(np.sum(input_img - cur_img, axis=-1))
print(f"difference shape = {difference.shape}")
print(f"min difference = {np.min(difference)}, max = {np.max(difference)}")


plt.figure(figsize=(10, 8))
cmap = 'viridis'
im = plt.imshow(difference, cmap=cmap, aspect='auto')
plt.title(title)
plt.colorbar(im, label='Values')

 #plt.xlabel('X Index')
# plt.ylabel('Y Index')
plt.tight_layout()
plt.savefig("difference_heatmap_not_bags2shoes_128.png")
plt.show()
# difference = (difference.clip(0, 1) * 255).astype(np.uint8)

# Image.fromarray(difference).save("difference.png")