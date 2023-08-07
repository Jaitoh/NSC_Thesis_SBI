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
    unravel_index,
    generate_permutations,
    get_len_seqC,
    choose_theta,
)
from utils.setup import clean_cache
from utils.set_seed import setup_seed
from train_nle.Dataset import probR_Comb_Dataset, BaseDataset


class x1pR_theta_Dataset(probR_Comb_Dataset):
    """
    dataset consisting of x and theta:
    - x consists: (seqC, 1, pR)
    - theta consists: (theta)

    ---
    original data:
    seqC            of shape:(1, M, _S, L)  - (1, 3, 3^n-1, 15)
    probR           of shape:(1, M, _S, T)  - (1, 3, 3^n-1, 500)
    theta           of shape:(T, 4)         - (500, 4)

    ---
    loaded into memory:
    seqC_normed     of shape (M*_S, L)    - (3*S, 15)
    probR           of shape (M*_S, T, 1) - (3*S, 500, 1)
    theta           of shape (T, 4)       - (500, 4)

    ===
    get item:
    x (seqC_normed[:-1], 1, pR)  of shape (15-1+1+1,) - (16)
    theta                   of shape (4) - (4)

    """

    def __init__(
        self,
        data_dir,
        num_chosen_theta=500,
        chosen_dur_list=[3, 9, 15],
        num_max_seqC_each_dur=[1, 1, 1],
        last_seqC_part=False,
        num_max_theta=500,
        theta_chosen_mode="random",
        print_info=False,
        config_theta=None,
    ):
        """
        theta_chosen_mode:
        'random'   - randomly choose 'num_chosen_theta' theta from all the theta in the set
        'first_80' - choose first 80% from 'num_chosen_theta' (normally as training set)
        'last_20'  - choose first 20% from 'num_chosen_theta' (normally as validation set)
        """
        super().__init__(
            data_dir=data_dir,
            num_max_theta=num_max_theta,
            num_chosen_theta=num_chosen_theta,
            chosen_dur_list=chosen_dur_list,
            num_max_seqC_each_dur=num_max_seqC_each_dur,
            last_seqC_part=last_seqC_part,
            theta_chosen_mode=theta_chosen_mode,
            config_theta=config_theta,
        )

        # set self.probR_all zero values to 1e-8
        self.probR_all[self.probR_all == 0] = 1e-8

        if print_info:
            print("".center(50, "="))
            print("[dataset info]")
            print(f"total # samples: {self.total_samples}")
            print(f"dur of {list(chosen_dur_list)}")
            print(f"part of {list(num_max_seqC_each_dur)} are chosen")

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
            print(f"loading one data time: {1000*(time.time()-start_loading_time):.2f}ms")
            print("".center(50, "="))

    def __getitem__(self, idx):
        """
        get a sample from the dataset of length M*S*T
        x: (seqC_normed[:-1], 1, pR)  of shape (15-1, pR) - (1, 500)
        theta: of shape (4) - (4)

        """
        # get the index
        seqC_idx, theta_idx = divmod(idx, self.T)

        # get x
        seqC = self.seqC_all[seqC_idx, 1:]
        probR = self.probR_all[seqC_idx, theta_idx]
        x = torch.cat((seqC, torch.tensor([1], dtype=seqC.dtype), probR), dim=0)

        # get theta
        theta = self.theta_all[theta_idx, :]

        return x, theta


# class x1pR_theta_Dataset_Complex_Partition(BaseDataset):

#     def __init__(
#         self,
#         data_dir,
#         chosen_dur_list=[3, 9, 15],
#         num_max_seqC_each_dur=[1, 1, 1],
#         num_max_theta=500,
#         config_theta=None,  # configurations for the theta values, norm / ignore_ss
#         training_set=True,
#         print_info=False,
#     ):
#         """
#         training_set:
#         True:   choose the first 90% of the seqCs, first  90% of the thetas
#         False:  choose the first 90% of the seqCs, last   10% of the thetas &
#                 choose the last  10% of the seqCs, first 100% of the thetas
#         """

#         super().__init__(
#             data_dir=data_dir,
#             chosen_dur_list=chosen_dur_list,
#             num_max_seqC_each_dur=num_max_seqC_each_dur,
#             num_max_theta=num_max_theta,
#             num_chosen_theta=num_max_theta,  # choose all the theta
#         )

#     def split_data(
#         self,
#         training_set=True,
#         chosen_dur_list=[3, 9, 15],
#         config_theta=None,
#     ):
#         # split the data accordingly
#         # self.seqC_all [M, S, 15]
#         # self.probR_all [M, S, T, 1]
#         # self.theta_all [T, 4]

#         train_seqC_idx, rest_seqC_idx = self._get_train_seqC_idx()
#         train_theta_idx, rest_theta_idx = self._get_train_theta_idx()

#         if training_set:
#             # slice the data
#             self.seqC_all = self.seqC_all[:, train_seqC_idx, :]  # [M, S, 15]
#             self.probR_all = self.probR_all[:, train_seqC_idx, train_theta_idx]  # [M, S, T]
#             self.theta_all = self.theta_all[train_theta_idx, :]  # [T, 4]
#             self.M, self.S, self.T = self.probR_all.shape
#             self.total_samples = self.M * self.S * self.T

#             # reshape the data
#             self.seqC_all = (
#                 torch.from_numpy(self.seqC_all)
#                 .reshape(self.M * self.S, self.L_seqC)
#                 .to(torch.float32)
#                 .contiguous()
#             )
#             self.probR_all = (
#                 torch.from_numpy(self.probR_all)
#                 .reshape(self.M * self.S, self.T)
#                 .unsqueeze(-1)
#                 .to(torch.float32)
#                 .contiguous()
#             )
#             self.theta_all = torch.from_numpy(self.theta_all).to(torch.float32).contiguous()
#             self.theta_all = process_theta_2D(  # normalize and processing of the theta values
#                 self.theta_all,
#                 ignore_ss=config_theta.ignore_ss,
#                 normalize_theta=config_theta.normalize,
#                 unnormed_prior_min=config_theta.prior_min,
#                 unnormed_prior_max=config_theta.prior_max,
#             )

#         else:
#             ...

#     def _get_train_seqC_idx(self):
#         starting_seqC_idx = 0
#         starting_seqC_idx_collection = []
#         ending_seqC_idx_collection = []
#         for S in self.S_each_dur:
#             starting_seqC_idx_collection.append(starting_seqC_idx)
#             num_seqC = np.floor(0.9 * S)
#             ending_seqC_idx = starting_seqC_idx + num_seqC
#             ending_seqC_idx_collection.append(ending_seqC_idx)
#             starting_seqC_idx += S

#         merged_idx = [
#             i
#             for start, end in zip(starting_seqC_idx_collection, ending_seqC_idx_collection)
#             for i in range(start, end)
#         ]

#         all_seqC_idx = set(range(self.S))
#         rest_idx = all_seqC_idx.difference(merged_idx)

#         return merged_idx, rest_idx

#     def _get_train_theta_idx(self):
#         full_list = list(range(self.T))
#         split_point = int(0.9 * self.T)

#         return full_list[:split_point], full_list[split_point:]

#     def __getitem__(self, idx):
#         return super().__getitem__(idx)
