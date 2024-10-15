import os

import torch
import numpy as np
from PIL import Image

import tqdm

from core.my_data_loader import load_dataset_h5py_no_transform

SEED = 0x000000
torch.manual_seed(SEED)
np.random.seed(SEED)

path_to_data = '../data'
DATASET1, DATASET1_PATH = 'outdoor', os.path.join(path_to_data, 'outdoor_128.hdf5')
DATASET2, DATASET2_PATH = 'church', os.path.join(path_to_data, 'church_128.hdf5')
IMG_SIZE = 128
batch_size = 32

path_to_save = "/cache/selikhanovych/ot/data"

modes = ["train", "test", "train", "test"]
datasets = ["outdoor", "outdoor", "church", "church"]
pair_name = 'outdoor2church'

if 0 > 1:
    train_loader_a, test_loader_a = load_dataset_h5py_no_transform(DATASET1_PATH,
                                                                   batch_size=batch_size)
    train_loader_b, test_loader_b = load_dataset_h5py_no_transform(DATASET2_PATH,
                                                                   batch_size=batch_size)

    loaders = [train_loader_a, test_loader_a, train_loader_b, test_loader_b]
    for i in range(4):
        loader = loaders[i]
        dataset = datasets[i]
        mode = modes[i]
        print(f"Dataset = {dataset}, mode = {mode}")
        index = 0
        path_to_folder = os.path.join(path_to_save, f"{pair_name}_{mode}", dataset)
        os.system(f"mkdir -p {path_to_folder}")
        for x, _ in tqdm.tqdm(loader):
            x_numpy = x.cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
            num_images_in_batch = x_numpy.shape[0]
            for j in range(num_images_in_batch):
                cur_image = Image.fromarray(x_numpy[j])
                cur_index = j + index
                cur_name = f"{cur_index}.png"
                cur_full_name = os.path.join(path_to_folder, cur_name)
                cur_image.save(cur_full_name)

            index += num_images_in_batch
        
for i in range(4):
    dataset = datasets[i]
    mode = modes[i]
    new_folder_name = os.path.join(path_to_save, f"{pair_name}_{mode}_{dataset}", "0")
    command1 = f"mkdir -p {new_folder_name}"
    os.system(command1)
    for j in range(10):
        path_to_folder = os.path.join(path_to_save, f"{pair_name}_{mode}", dataset, f"*{j}.png")
        command2 = f"cp -r {path_to_folder} {new_folder_name}"
        os.system(command2)
