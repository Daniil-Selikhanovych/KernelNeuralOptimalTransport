import os
import glob

root = "/trinity/home/daniil.selikhanovych/data/img_align_celeba"

path_to_celeba_attr = os.path.join(root, "..", "list_attr_celeba.txt")
with open(path_to_celeba_attr, 'r') as f:
    lines = f.readlines()[2:]

female_imgs = [lines[i].replace('  ', ' ').split(' ')[0] for i in list(range(len(lines)))
                    if lines[i].replace('  ', ' ').split(' ')[21] == '-1']
male_imgs = [lines[i].replace('  ', ' ').split(' ')[0] for i in list(range(len(lines)))
                    if lines[i].replace('  ', ' ').split(' ')[21] != '-1']

female_paths = sorted([os.path.join(root, img) for img in female_imgs])
male_paths = sorted([os.path.join(root, img) for img in male_imgs])

all_images = sorted(glob.glob(os.path.join(root, "*.jpg")))
num_all_paths = len(all_images)
num_female_paths = len(female_imgs)
num_male_paths = len(male_imgs)
assert num_male_paths + num_female_paths == num_all_paths

print(f"num celeba male paths = {num_male_paths}, num celeba female = {num_female_paths}, num all images = {num_all_paths}")