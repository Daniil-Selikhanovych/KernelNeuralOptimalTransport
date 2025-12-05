import os, sys
sys.path.append("..")

# import matplotlib
import numpy as np
# import matplotlib.pyplot as plt
# %matplotlib inline 

import numpy as np
import torch
import torch.nn as nn
# import torchvision
# import gc

# from src.tools import unfreeze, freeze
from src.tools import load_dataset, get_loader_stats

from copy import deepcopy
import json

from tqdm import tqdm
# from IPython.display import clear_output

# This needed to use dataloaders for some datasets
# from PIL import PngImagePlugin
# LARGE_ENOUGH_NUMBER = 100
# PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

DEVICE_ID = 0

path_to_data = "/trinity/home/daniil.selikhanovych/my_thesis/datasets"
path_to_shoes = os.path.join(path_to_data, "shoes_128.hdf5")
path_to_shoes_64 = os.path.join(path_to_data, "shoes_64.hdf5")
path_to_dtd = os.path.join(path_to_data, "dtd/images")

DATASET_LIST = [
#     ('handbag', '../../data/handbag_128.hdf5', 64, 1),
# ('shoes', path_to_shoes, 64, 1),
# ('shoes', path_to_shoes_64, 64, 1),
#     ('handbag', '../../data/handbag_128.hdf5', 128, 1),
#     ('shoes', '../../data/shoes_128.hdf5', 64, 1),
   #  ('shoes', '../../data/shoes_128.hdf5', 128, 1),
   #('shoes', path_to_shoes, 128, 1),
    #  ('dtd', path_to_dtd, 64, 50),
     ('celeba_female', '../../data/img_align_celeba_all', 64, 1),
#     ('aligned_anime_faces', '../../data/aligned_anime_faces', 128, 1),
]

assert torch.cuda.is_available()
torch.cuda.set_device(f'cuda:{DEVICE_ID}')

for DATASET, DATASET_PATH, IMG_SIZE, N_EPOCHS in tqdm(DATASET_LIST):
    print('Processing {}'.format(DATASET))
    sampler, test_sampler = load_dataset(DATASET, DATASET_PATH, img_size=IMG_SIZE)
    print('Dataset {} loaded'.format(DATASET))

    mu, sigma = get_loader_stats(test_sampler.loader, n_epochs=N_EPOCHS)
    print('Trace of sigma: {}'.format(np.trace(sigma)))
    stats = {'mu' : mu.tolist(), 'sigma' : sigma.tolist()}
    print('Stats computed')

    filename = os.path.join(path_to_data, '{}_{}_test_2.json'.format(DATASET, IMG_SIZE))
    with open(filename, 'w') as fp:
        json.dump(stats, fp)
    print('States saved to {}'.format(filename))