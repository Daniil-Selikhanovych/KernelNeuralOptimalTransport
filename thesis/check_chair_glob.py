import os
import glob

def check_dir(directory):
    extensions = ["*.jpg", "*.png", "*.PNG", "*.JPG", "*.JPEG", "*.jpeg"]
    num_files = 0
    for ext in extensions:
        file_ext = sorted(glob.glob(os.path.join(directory, ext)))
        num_files += len(file_ext)
        
    return num_files

path_to_chairs = "/trinity/home/daniil.selikhanovych/my_thesis/datasets/houzz/chairs"

all_subdirs = sorted(glob.glob(os.path.join(path_to_chairs, "*")))

all_chairs_file = 0
for directory in all_subdirs:
    num_files = check_dir(directory)
    all_chairs_file += num_files
    
print(f"all_chairs_file = {all_chairs_file}")