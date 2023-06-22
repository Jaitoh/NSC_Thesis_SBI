""" Dataset-features.py
"""

import time
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class Feature_Dataset(Dataset):
    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.config = config
        self.data_path = config.dataset.data_path

        #

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


data_path = "/mnt/data/dataset/feature-L0-Eset0-100sets-T500-C100.h5"
# load dataset into memory
f = h5py.File(data_path, "r")
sets = list(f.keys())

# 90% sets for training, 10% for testing
train_set_names = sets[:90]
test_set_names = sets[90:]
val_set_names = sets

train_set_share = [0, 0.9]
val_set_share = [0.9, 1.0]
test_set_share = [0, 0.9]

concatenate_feature_types = [1, 2, 3, 4, 5]

# === above is predefined values ===
len_feature_each_type = [
    f[train_set_names[0]][f"feature_{i}"].shape[-1] for i in range(1, 6)
]
n_features = sum(len_feature_each_type)
# with h5py.File(data_path, "r") as f:
T, C, M = f[train_set_names[0]]["feature_1"].shape[:-1]
len_theta = f[train_set_names[0]]["theta"].shape[-1]

# prepare training data
n_sets = len(train_set_names)
n_T = int(T * (train_set_share[1] - train_set_share[0]))

x_train = torch.empty((n_sets, n_T, C, M, n_features))
theta_train = torch.empty((n_sets, n_T, len_theta))

for idx_set, set_name in enumerate(train_set_names):
    pass  # TODO: finish this

# concatenated_features = torch.cat((feature1, feature2, feature4, feature5), dim=3)
