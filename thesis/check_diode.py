import os
import glob
import pandas as pd

path_to_train_indoor = "/trinity/home/daniil.selikhanovych/my_thesis/datasets/diode/data_list"
all_csv = sorted(glob.glob(os.path.join(path_to_train_indoor, "*.csv")))

for csv_file in all_csv:
    csv_name = os.path.basename(csv_file)
    pd_file = pd.read_csv(csv_file, header=None)
    print(f"file {csv_name} = {len(pd_file[0])}")