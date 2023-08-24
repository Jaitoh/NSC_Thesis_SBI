import time
import h5py
import torch
import numpy as np
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import gc
from pathlib import Path

# from dataset.data_process import process_x_seqC_part
import sys

NSC_DIR = Path(__file__).resolve().parent.parent.parent.parent.as_posix()  # NSC dir
sys.path.append(f"{NSC_DIR}/codes/src")

from dataset.data_process import process_x_seqC_part
from utils.dataset.dataset import (
    process_theta_2D,
    process_theta_3D,
    unravel_index,
    generate_permutations,
    get_len_seqC,
    choose_theta,
)
from utils.setup import clean_cache
from utils.set_seed import setup_seed
from utils.dataset.dataset import pR2cR_acc
from utils.setup import adapt_path


# seqC_process = ("norm",)
# summary_type = ("0",)
"""
load into memory:
    seqC: (nSets, D, M, S, 15)
    probR: (nSets, D, M, S, T, 1)
    theta: (nSets, T, 4)
"""

dataset_path = adapt_path("/home/wehe/tmp/NSC/data/dataset/dataset-L0-Eset0-100sets-T500.h5")
f = h5py.File(dataset_path, "r")
sets = list(f.keys())[:10]


chosen_set_names = sets[: int(len(sets) * 0.9)]

f[sets[0]].keys()  # ["probR", "seqC", "theta"]
f[sets[0]]["probR"].shape  # (D, M, S, T, 1)
f[sets[0]]["seqC"].shape  # (D, M, S, 15)
f[sets[0]]["theta"].shape  # (T, 4)


# use the dataset from npe for nle
class BaseDataset(Dataset):
    def __init__(
        self,
        data_path="/home/wehe/tmp/NSC/data/dataset/dataset-L0-Eset0-100sets-T500-copy.h5",
        chosen_set_names=["set_0", "set_1"],
        config_theta=None,
    ):
        super().__init__()

        """Loading the high-dimensional sets into memory
        ---
        load into memory:
            seqC: (nSets, D, M, S, 15)
            probR: (nSets, D, M, S, T, 1)
            theta: (nSets, T, 4)
        """
        # start_loading_time = time.time()
        print("start loading data into MEM ... ")
        data_path = adapt_path(data_path)
        # define the final shape of the data
        self.nSets = len(chosen_set_names)

        with h5py.File(data_path, "r") as f:
            sets = list(f.keys())
            self.D, self.M, self.S, self.T = f[sets[0]]["probR"].shape[:4]
            self.L_seqC = f[sets[0]]["seqC"].shape[-1]

        self.seqC_all = np.zeros((self.nSets, self.D, self.M, self.S, self.L_seqC))
        self.probR_all = np.zeros((self.nSets, self.D, self.M, self.S, self.T, 1))
        self.theta_all = np.zeros((self.nSets, self.T, 4))

        self.total_samples = self.nSets * self.D * self.M * self.S * self.T

        counter = 0
        with h5py.File(data_path, "r") as f:
            for set_idx, set_name in enumerate(chosen_set_names):
                print(counter, end=" ")  # if counter % 2 == 0 else None

                # load data # libver="latest",swmr=True
                seqC_data = process_x_seqC_part(
                    seqC=f[set_name]["seqC"][:],
                    seqC_process="norm",  # 'norm' or 'summary'
                    nan2num=-1,
                    summary_type=0,  # 0 or 1
                )
                # seqC_data (D, M, S, L)

                probR_data = f[set_name]["probR"][:]
                # probR_data (D, M, S, T, 1)

                theta_data = f[set_name]["theta"][:]
                # theta_data (T, 4)

                self.seqC_all[set_idx, ...] = seqC_data
                self.probR_all[set_idx, ...] = probR_data
                self.theta_all[set_idx, ...] = theta_data

                del seqC_data, probR_data, theta_data
                counter += 1

        self.theta_all = process_theta_3D(  # normalize and processing of the theta values
            self.theta_all,
            ignore_ss=config_theta.ignore_ss,
            normalize_theta=config_theta.normalize,
            unnormed_prior_min=config_theta.prior_min,
            unnormed_prior_max=config_theta.prior_max,
        )

        # set self.probR_all zero values to 1e-8
        self.probR_all[self.probR_all == 0] = 1e-8

    def __len__(self):
        return self.total_samples


class x1pR_theta_Dataset(BaseDataset):
    """
    dataset consisting of x and theta:
    - x consists: (seqC, 1, pR)
    - theta consists: (theta)

    ===
    get item:
    x (seqC_normed[:-1], 1, pR)  of shape (15-1+1+1,) - (16)
    theta                   of shape (4) - (4)

    """

    def __init__(
        self,
        data_path="/home/wehe/tmp/NSC/data/dataset/dataset-L0-Eset0-100sets-T500-copy.h5",
        chosen_set_names=["set_0", "set_1"],
        config_theta=None,
        print_info=True,
    ):
        """
        theta_chosen_mode:
        'random'   - randomly choose 'num_chosen_theta' theta from all the theta in the set
        'first_80' - choose first 80% from 'num_chosen_theta' (normally as training set)
        'last_20'  - choose first 20% from 'num_chosen_theta' (normally as validation set)
        """
        super().__init__(
            data_path=data_path,
            chosen_set_names=chosen_set_names,
            config_theta=config_theta,
        )

        self.seqC_all = torch.from_numpy(self.seqC_all).to(torch.float32).contiguous()
        self.probR_all = torch.from_numpy(self.probR_all).to(torch.float32).contiguous()
        self.theta_all = torch.from_numpy(self.theta_all).to(torch.float32).contiguous()

        # index
        indices = torch.arange(self.total_samples)
        (
            self.set_idxs,
            self.D_idxs,
            self.M_idxs,
            self.S_idxs,
            self.T_idxs,
        ) = unravel_index(indices, (self.nSets, self.D, self.M, self.S, self.T))

        if print_info:
            print("".center(50, "="))
            print("[dataset info]")
            print(f"total # samples: {self.total_samples}")
            print(f"chosen sets {chosen_set_names}")

            print("".center(50, "-"))
            print("shapes:")
            print(f"[seqC] shape: {self.seqC_all.shape}")
            print(f"[theta] shape: {self.theta_all.shape}")
            print(f"[probR] shape: {self.probR_all.shape}")

            print("".center(50, "-"))
            print("example:")
            start_loading_time = time.time()
            print(f"[x] e.g. {self.__getitem__(0)[0]}")
            print(f"[theta] e.g. {self.__getitem__(0)[1]}")
            print(f"loading one data time: {500*(time.time()-start_loading_time):.2f}ms")
            print("".center(50, "="))

    def __getitem__(self, idx):
        """
        get a sample from the dataset of length M*S*T
        x: (seqC_normed[:-1], 1, pR)  of shape (15-1, pR) - (1, 500)
        theta: of shape (4) - (4)

        """
        # get the index
        set_idx, D_idx, M_idx, S_idx, T_idx = (
            self.set_idxs[idx],
            self.D_idxs[idx],
            self.M_idxs[idx],
            self.S_idxs[idx],
            self.T_idxs[idx],
        )

        # get x
        seqC = self.seqC_all[set_idx, D_idx, M_idx, S_idx, 1:]
        probR = self.probR_all[set_idx, D_idx, M_idx, S_idx, T_idx, :]
        x = torch.cat((seqC, torch.tensor([1], dtype=seqC.dtype), probR), dim=0)

        # get theta
        theta = self.theta_all[set_idx, T_idx, :]

        return x, theta


if __name__ == "__main__":
    setup_seed(100)

    # turn config_theta into a class
    class config_theta:
        def __init__(self):
            self.ignore_ss = False
            self.normalize = True
            self.prior_min = [-2.5, 0, 0, -11]
            self.prior_max = [2.5, 77, 18, 10]

    config_theta = config_theta()

    Dataset = x1pR_theta_Dataset(
        data_path="/home/wehe/tmp/NSC/data/dataset/dataset-L0-Eset0-100sets-T500-copy.h5",
        chosen_set_names=["set_0", "set_1"],
        config_theta=config_theta,
    )
    x, theta = Dataset[0]
