import os

import torch
import numpy as np
from PIL import Image

import tqdm

from core.my_data_loader import load_dataset_celeba_no_transform

SEED = 0x000000
torch.manual_seed(SEED)
np.random.seed(SEED)

DATASET1, DATASET1_PATH = 'celeba_male', './data/img_align_celeba'
DATASET2, DATASET2_PATH = 'celeba_female', './data/img_align_celeba'
IMG_SIZE = 64
batch_size = 32

# path_to_save = "/cache/selikhanovych/ot/data"
path_to_save = "/cache/selikhanovych/OT_competitors/stargan-v2/data"

train_loader_a, test_loader_a = load_dataset_celeba_no_transform(DATASET1, DATASET1_PATH,
                                                                 batch_size=batch_size)
train_loader_b, test_loader_b = load_dataset_celeba_no_transform(DATASET2, DATASET2_PATH,
                                                                 batch_size=batch_size)

loaders = [train_loader_a, test_loader_a, train_loader_b, test_loader_b]
modes = ["train", "test", "train", "test"]
sexes = ["male", "male", "female", "female"]

for i in range(4):
    loader = loaders[i]
    sex = sexes[i]
    mode = modes[i]
    print(f"Sex = {sex}, mode = {mode}")
    index = 0
    path_to_folder = os.path.join(path_to_save, f"celeba_{mode}", sex)
    os.system(f"mkdir -p {path_to_folder}")
    for x, _ in tqdm.tqdm(loader):
        x_numpy = (x * 255).cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
        num_images_in_batch = x_numpy.shape[0]
        for j in range(num_images_in_batch):
            cur_image = Image.fromarray(x_numpy[j])
            cur_index = j + index
            cur_name = f"{cur_index}.png"
            cur_full_name = os.path.join(path_to_folder, cur_name)
            cur_image.save(cur_full_name)

        index += num_images_in_batch

