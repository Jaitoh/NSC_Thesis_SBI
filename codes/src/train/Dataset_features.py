""" Dataset-features.py
"""
import gc
import time
import h5py
import torch
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys

sys.path.append("../../src")
sys.path.append("./src")
from utils.dataset import unravel_index


class Feature_Dataset(Dataset):
    def __init__(
        self,
        data_path,
        set_names,
        set_T_part=[0, 0.9],
        concatenate_feature_types=[1, 3, 4, 5],
        concatenate_along_M=True,
    ):
        """
        The resulting dataset
            (concatenate_along_M=True):
                x    : (n_sets, n_T, C, M * n_features, 1)
                theta: (n_sets, n_T, n_theta)

            (concatenate_along_M=False):
                x    : (n_sets, n_T, C, n_features, M)
                theta: (n_sets, n_T, n_theta)

        max idxing length: n_sets * n_T * C

        idxed data shape:
            (concatenate_along_M=True):
                x    : (M * n_features, 1)
                theta: (n_theta)

            (concatenate_along_M=False):
                x    : (n_features, M)
                theta: (n_theta)
        """

        super().__init__()

        with h5py.File(data_path, "r") as f:
            # get the length of each feature type
            len_feature_each_type = [
                f[set_names[0]][f"feature_{i}"].shape[-1]
                for i in concatenate_feature_types
            ]
            print(f"{len_feature_each_type=}")

            self.len_feature_each_type = len_feature_each_type
            n_features = sum(len_feature_each_type)
            n_theta = f[set_names[0]]["theta"].shape[-1]
            T, C, M = f[set_names[0]]["feature_1"].shape[:-1]
            n_sets = len(set_names)
            n_T = int(T * (set_T_part[1] - set_T_part[0]))
            range_T = [int(set_T_part[0] * T), int(set_T_part[1] * T)]

            self.total_samples = n_sets * n_T * C
            self.x = torch.empty((n_sets, n_T, C, M, n_features))
            self.theta = torch.empty((n_sets, n_T, n_theta))

            print(f"loading {n_sets}sets T{(set_T_part[1]-set_T_part[0])*100:.2f}%")
            for idx_set, set_name in tqdm(
                enumerate(set_names), total=n_sets, miniters=n_sets // 5
            ):
                # for each selected set
                # extract feature of partial T from range_T[0] to range_T[1]
                # concatenate selected different features along the last dimension
                chosen_feature_data = [
                    torch.from_numpy(
                        f[set_name][f"feature_{i}"][range_T[0] : range_T[1], :, :, :]
                    )
                    for i in concatenate_feature_types
                ]
                concatenated_features = torch.cat(chosen_feature_data, dim=-1)
                self.x[idx_set] = concatenated_features

                # extract theta of partial T from range_T[0] to range_T[1]
                self.theta[idx_set] = torch.from_numpy(
                    f[set_name]["theta"][range_T[0] : range_T[1], :]
                )

        if concatenate_along_M:
            self.x = self.x.view(n_sets, n_T, C, M * n_features)
            self.x = self.x.unsqueeze(-1)  # (n_sets, n_T, C, M * n_features, 1)
        else:
            self.x = self.x.transpose(-1, -2)  # (n_sets, n_T, C, n_features, M)

        print(
            f"dataset info: ==> {self.total_samples=} => {self.x.shape=} {self.theta.shape=}"
        )

        # get the idxs for each sample
        indices = torch.arange(self.total_samples)
        self.set_idxs, self.T_idxs, self.C_idxs = unravel_index(indices, (n_sets, n_T, C))

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        set_idx, T_idx, C_idx = self.set_idxs[idx], self.T_idxs[idx], self.C_idxs[idx]
        return self.x[set_idx, T_idx, C_idx], self.theta[set_idx, T_idx]


def main(concatenate_feature_types=[3]):
    data_path = "/mnt/data/dataset/feature-L0-Eset0-100sets-T500-C100.h5"
    data_path = "/home/ubuntu/tmp/NSC/data/dataset/feature-L0-Eset0-100sets-T500-C100.h5"
    # data_path = "/home/wehe/tmp/NSC/data/dataset/feature-L0-Eset0-100sets-T500-C100.h5"
    f = h5py.File(data_path, "r")
    sets = list(f.keys())
    f.close()

    # if Feature exists, delete it
    if "Feature" in locals():
        del Feature
        print("deleted Feature")
        gc.collect()

    # 90% sets for training, 10% for testing
    train_set_names = sets[:90]
    test_set_names = sets[90:]
    val_set_names = sets

    train_set_T_part = [0, 0.9]
    val_set_T_part = [0.9, 1.0]
    test_set_T_part = [0, 0.9]
    Feature = Feature_Dataset(
        data_path=data_path,
        set_names=train_set_names,
        set_T_part=train_set_T_part,
        concatenate_feature_types=concatenate_feature_types,
    )

    len_feature_each_type = Feature.len_feature_each_type

    # load one sample
    plt.figure(figsize=(10, 5))
    for idx in [0, 10, 101, 111, 45300]:
        x, theta = Feature[idx]
        print(
            f"{idx=:7} {theta=} {Feature.set_idxs[idx], Feature.T_idxs[idx], Feature.C_idxs[idx]} {x.shape=} {theta.shape=}"
        )  # {x.shape=} {theta.shape=}
        plt.plot(x, label=f"{idx=}")
    # set the legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.grid(alpha=0.2)
    plt.title(f"{concatenate_feature_types=}\n{len_feature_each_type=}")
    plt.xlabel("feature index")
    plt.ylabel("feature value")
    # vertical line
    value_accumulate = 0
    for i in range(3):
        for value in len_feature_each_type:
            color = "k" if value == len_feature_each_type[-1] else "grey"
            plt.axvline(
                value_accumulate := value_accumulate + value, color=color, linestyle="--"
            )

    # set boarder width to 3
    for axis in ["top", "bottom", "left", "right"]:
        plt.gca().spines[axis].set_linewidth(3)
    plt.show()

    # get a dataloader
    loader_kwargs = {
        "batch_size": 32,
        "drop_last": True,
        "pin_memory": False,
        "num_workers": 4,
        "prefetch_factor": 3,
    }
    print(f"{loader_kwargs=}")

    g = torch.Generator()
    g.manual_seed(100)

    train_dataloader = data.DataLoader(Feature, generator=g, **loader_kwargs)
    x_batch, theta_batch = next(iter(train_dataloader))
    print(f"{x_batch.shape=} {theta_batch.shape=}")


if __name__ == "__main__":
    concatenate_feature_types = [1]
    main(concatenate_feature_types=concatenate_feature_types)
    concatenate_feature_types = [2]
    main(concatenate_feature_types=concatenate_feature_types)
    concatenate_feature_types = [3]
    main(concatenate_feature_types=concatenate_feature_types)
    concatenate_feature_types = [4]
    main(concatenate_feature_types=concatenate_feature_types)
    concatenate_feature_types = [5]
    main(concatenate_feature_types=concatenate_feature_types)
    concatenate_feature_types = [1, 2]
    main(concatenate_feature_types=concatenate_feature_types)
    concatenate_feature_types = [1, 3]
    main(concatenate_feature_types=concatenate_feature_types)
    concatenate_feature_types = [3, 4]
    main(concatenate_feature_types=concatenate_feature_types)
    concatenate_feature_types = [1, 2, 3, 4, 5]
    main(concatenate_feature_types=concatenate_feature_types)
