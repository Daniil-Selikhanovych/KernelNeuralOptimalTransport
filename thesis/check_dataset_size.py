import os
import h5py
import numpy as np

datasets = ["handbag_64.hdf5", "outdoor_128.hdf5", "church_128.hdf5"]

path_to_data = "/trinity/home/daniil.selikhanovych/my_thesis/datasets"
# path_to_shoes = "/trinity/home/daniil.selikhanovych/my_thesis/datasets/shoes_64.hdf5"
# path_to_handbags = "/trinity/home/daniil.selikhanovych/my_thesis/datasets/handbag_64.hdf5"
# path_to_handbags = "/trinity/home/daniil.selikhanovych/my_thesis/datasets/handbag_64.hdf5"
for data_name in datasets:
    path_to_dataset = os.path.join(path_to_data, data_name)
    print(f"data_name = {data_name}")
    with h5py.File(path_to_dataset, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = list(f[a_group_key])

    data = np.array(data)
    print(f"data size = {data.shape}")
