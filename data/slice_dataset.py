""" Slice original dataset to 500
"""
import numpy as np
import h5py
from tqdm import tqdm

DATA_PATH = "/mnt/data/dataset/dataset_L0_exp_set_0.h5"
NEW_DATA_PATH = "/mnt/data/dataset/dataset_L0_exp_set_0_T500.h5"

# load data
f_src = h5py.File(DATA_PATH, "r")
# create new h5 file
f_dest = h5py.File(NEW_DATA_PATH, "a")
sets = list(f_src.keys())
print(f_src[sets[0]].keys())
print(f_src[sets[0]]["seqC"].shape)
print(f_src[sets[0]]["probR"].shape)
print(f_src[sets[0]]["theta"].shape)

for group_name, group in tqdm(f_src.items()):
    # copy and slice data to new dataset f_new
    dest_group = f_dest.create_group(group_name)
    group.copy("seqC", dest_group)

    theta_data = group["theta"][:500, :]
    dest_group.create_dataset("theta", data=theta_data)

    probR_data = group["probR"][:, :, :, :500, :]
    dest_group.create_dataset("probR", data=probR_data)


print(f_dest["set_0"]["seqC"].shape)
print(f_dest["set_0"]["probR"].shape)
print(f_dest["set_0"]["theta"].shape)

assert (
    np.sum(
        np.nan_to_num(f_dest["set_0"]["seqC"][:])
        - np.nan_to_num(f_src["set_0"]["seqC"][:])
    )
    == 0
)
assert (
    np.sum(f_dest["set_11"]["probR"] - f_src["set_0"]["probR"][:, :, :, :500, :]) == 0
)
assert np.sum(f_dest["set_1"]["theta"] - f_src["set_0"]["theta"][:500, :]) == 0

f_src.close()
f_dest.close()
