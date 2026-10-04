import os
import glob
import shutil

path = "/trinity/home/daniil.selikhanovych/data/img_align_celeba_all/img_align_celeba"

num_train_samples_a = 90000
num_train_samples_b = 90000

all_images = sorted(glob.glob(os.path.join(path, "*")))

path_to_save = "/trinity/home/daniil.selikhanovych/OptimalTransportModeling/data/celeba"
images_a = all_images[:num_train_samples_a]
images_b = all_images[num_train_samples_a:num_train_samples_a+num_train_samples_b]
images_c = all_images[num_train_samples_a+num_train_samples_b:]

path_to_save_a = os.path.join(path_to_save, "trainA")
path_to_save_b = os.path.join(path_to_save, "trainB")
path_to_save_c = os.path.join(path_to_save, "testA")

os.makedirs(path_to_save_a, exist_ok=True)
os.makedirs(path_to_save_b, exist_ok=True)
os.makedirs(path_to_save_c, exist_ok=True)

for path in images_a:
    image_basename = os.path.basename(path)
    path_to_save = os.path.join(path_to_save_a, image_basename)
    shutil.copy(path, path_to_save)
    
for path in images_b:
    image_basename = os.path.basename(path)
    path_to_save = os.path.join(path_to_save_b, image_basename)
    shutil.copy(path, path_to_save)
    
for path in images_c:
    image_basename = os.path.basename(path)
    path_to_save = os.path.join(path_to_save_c, image_basename)
    shutil.copy(path, path_to_save)
