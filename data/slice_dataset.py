""" Slice original dataset to 500
"""
import numpy as np
import h5py
from tqdm import tqdm
import os

DATA_PATH = "/home/wehe/tmp/NSC/data/dataset/dataset_L0_exp_0_set100_T5000.h5"
NEW_DATA_PATH = "/home/wehe/tmp/NSC/data/dataset/dataset_L0_exp_0_set100_T500.h5"

# remove NEW_DATA_PATH if exists
if os.path.exists(NEW_DATA_PATH):
    os.remove(NEW_DATA_PATH)

# load data
with h5py.File(DATA_PATH, "r") as f_src:
    with h5py.File(NEW_DATA_PATH, "a") as f_dest:
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
        ), "seqC not equal"
        assert (
            np.sum(
                f_dest["set_11"]["probR"] - f_src["set_11"]["probR"][:, :, :, :500, :]
            )
            == 0
        ), "probR not equal"
        assert (
            np.sum(f_dest["set_1"]["theta"] - f_src["set_1"]["theta"][:500, :]) == 0
        ), "theta not equal"
