import os
import h5py 
import numpy as np
from PIL import Image
from tqdm import tqdm

path_to_data = "/trinity/home/daniil.selikhanovych/my_thesis/datasets"
path_to_shoes = os.path.join(path_to_data, "shoes_128.hdf5")
path_to_dtd = os.path.join(path_to_data, "dtd/images")
path_to_handbags = os.path.join(path_to_data, "handbag_128.hdf5")

# with h5py.File(path_to_handbags, "r") as f:
with h5py.File(path_to_shoes, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    print(f"a_group_key = {a_group_key}")
    # Get the data
    data = list(f[a_group_key])
    
data = np.array(data)
print(f"data.shape = {data.shape}, dtype = {data.dtype}")

# path_to_save = os.path.join(path_to_data, "handbag_128_png")
path_to_save = os.path.join(path_to_data, "shoes_128_png")
os.makedirs(path_to_save, exist_ok=True)

num_images = data.shape[0]
for i in tqdm(range(num_images)):
    cur_img = data[i]
    img_name = f"{i}.png"
    path_to_save_cur = os.path.join(path_to_save, img_name)
    Image.fromarray(cur_img).save(path_to_save_cur)
    