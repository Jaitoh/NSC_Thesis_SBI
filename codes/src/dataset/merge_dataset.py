import os
import h5py
import argparse
from tqdm import tqdm
from pathlib import Path

parser = argparse.ArgumentParser(description='pipeline for sbi')
parser.add_argument('--data_dir', type=str, default="/home/wehe/scratch/data/dataset",
                    help="simulated data store/load dir")
args = parser.parse_args()

# list the files in the data directory using python
# data_dir = '../../data/dataset'
# data_dir = '/home/wehe/scratch/data/dataset'
data_dir = args.data_dir
files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

data_paths = []
data_file_idxs = []
for data_file in files:
    data_path = str(Path(data_dir) / data_file)
    data_paths.append(data_path)
    data_file_idxs.append(int(data_path.split('_part_')[-1].split('.')[0]))
# sort the data paths according to the file name
# data_paths.sort(key=lambda x: x.split('_part_')[-1].split('.')[0])

sorted_indices = sorted(range(len(data_file_idxs)), key=lambda i: data_file_idxs[i])

# create a new h5 file to store the combined dataset
merged_data_path = Path(data_dir) / 'dataset_L0_exp_set_0.h5'

print(f'merged data_path {merged_data_path}')
with h5py.File(merged_data_path, 'w') as merged_data_file:
    for i in tqdm(sorted_indices):
        data_path, data_file_idx = data_paths[i], data_file_idxs[i]
        print(f'adding {data_path} to [set_{data_file_idx}]')
        data_file = h5py.File(data_path, 'r')
        # copy the data group from the data file to the merged data file
        set_group = merged_data_file.create_group(f'set_{data_file_idx}')
        data_file.copy('data/seqC', set_group)
        data_file.copy('data/probR', set_group)
        data_file.copy('data/theta', set_group)
        data_file.close()
        merged_data_file.flush()

    print(f'\nmerged file has keys: {list(merged_data_file.keys())}')    
    print(f"\nin one set_0 it has keys: {list(merged_data_file['set_0'].keys())}")
    print(f"seqC has a shape of: {merged_data_file['set_0']['seqC'].shape}")
    print(f"theta has a shape of: {merged_data_file['set_0']['theta'].shape}")
    print(f"probR has a shape of: {merged_data_file['set_0']['probR'].shape}")

# merged_data_file.close()