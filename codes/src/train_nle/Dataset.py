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


class probR_Comb_Dataset(Dataset):
    def __init__(
        self,
        data_dir,
        num_chosen_theta=500,
        chosen_dur_list=[3, 9, 15],
        part_each_dur=[1, 1, 1],
        last_part=False,
        num_max_theta=500,
        theta_chosen_mode="random",
        print_info=False,
        config_theta=None,
    ):
        super().__init__()

        """Loading the high-dimensional sets into memory
        ---
        original data:
        seqC            of shape:(1, M, _S, L)  - (1, 3, 3^n-1, 15)
        probR           of shape:(1, M, _S, T)   - (1, 3, 3^n-1, 500)
        theta           of shape:(T, 4)         - (500, 4)
        
        ---
        loaded into memory:
        seqC_normed     of shape (M*_S, L)    - (3*S, 15)
        probR           of shape (M*_S, T, 1) - (3*S, 500, 1)
        theta           of shape (T, 4)       - (500, 4)
        
        ===
        get item:
        seqC_normed     of shape (L) - (15-1)
        probR           of shape (1) - (1)
        theta           of shape (4) - (4)
        
        
        theta_chosen_mode:
            'random'   - randomly choose 'num_chosen_theta' theta from all the theta in the set
            'first_80' - choose first 80% from 'num_chosen_theta' (normally as training set)
            'last_20'  - choose first 20% from 'num_chosen_theta' (normally as validation set)
        """
        start_loading_time = time.time()
        print("start loading data into MEM ... ", end="")
        self.num_chosen_theta = num_chosen_theta

        # define the final shape of the data
        chosen_theta_idx, num_chosen_theta = choose_theta(
            num_chosen_theta, num_max_theta, theta_chosen_mode
        )
        self.T = num_chosen_theta

        self.S = 0
        for dur, part in zip(chosen_dur_list, part_each_dur):
            f = h5py.File(Path(data_dir) / f"dataset-comb-dur{dur}-T500.h5", "r")
            _, self.M, S, self.L_seqC = f["seqC"].shape  # (1, M, S, L)
            _S = round(S * part)
            self.S += _S
            self.theta_all = f["theta"][chosen_theta_idx, :]
            f.close()
        # self.theta_all = np.empty((self.T, 4))
        self.seqC_all = np.empty((self.M, self.S, self.L_seqC))
        self.probR_all = np.empty((self.M, self.S, self.T))

        self.total_samples = self.M * self.S * self.T

        S_cnt, counter = 0, 0
        for dur, part in zip(chosen_dur_list, part_each_dur):
            print(counter, end=" ")  # if counter % 2 == 0 else None

            # load data # libver="latest",swmr=True
            f = h5py.File(Path(data_dir) / f"dataset-comb-dur{dur}-T500.h5", "r")
            seqC_data = self._get_seqC_data(f, part, last_part=last_part)
            # seqC_data (M, _S, L)

            _S = seqC_data.shape[1]
            probR_data = (
                f["probR"][:, :, -_S:, chosen_theta_idx][0]
                if last_part
                else f["probR"][:, :, :_S, chosen_theta_idx][0]
            )
            # probR_data (M, _S, T)
            f.close()

            self.seqC_all[:, S_cnt : (S_cnt + _S), :] = seqC_data
            self.probR_all[:, S_cnt : (S_cnt + _S), :] = probR_data

            del seqC_data, probR_data
            counter += 1
            S_cnt += _S

        # convert to tensor and reshape
        self.seqC_all = (
            torch.from_numpy(self.seqC_all)
            .reshape(self.M * self.S, self.L_seqC)
            .to(torch.float32)
            .contiguous()
        )
        self.probR_all = (
            torch.from_numpy(self.probR_all)
            .reshape(self.M * self.S, self.T)
            .unsqueeze(-1)
            .to(torch.float32)
            .contiguous()
        )

        # process theta [nSets, TC, 4]
        self.theta_all = torch.from_numpy(self.theta_all).to(torch.float32).contiguous()
        self.theta_all = (
            process_theta_2D(  # normalize and processing of the theta values
                self.theta_all,
                ignore_ss=config_theta.ignore_ss,
                normalize_theta=config_theta.normalize,
                unnormed_prior_min=config_theta.prior_min,
                unnormed_prior_max=config_theta.prior_max,
            )
        )

        if print_info:
            self._print_info(chosen_dur_list, part_each_dur, start_loading_time)

    def _get_seqC_data(self, f, part, last_part=False):
        seqC_shape = f["seqC"].shape  # seqC: (1, M, S, L)
        S = seqC_shape[2]
        S_part = round(S * part)
        # take partial of S
        seqC = (
            f["seqC"][:, :, -S_part:, :][0]
            if last_part
            else f["seqC"][:, :, :S_part, :][0]
        )
        # seqC: (M, S, L)

        return process_x_seqC_part(
            seqC=seqC,
            seqC_process="norm",  # 'norm' or 'summary'
            nan2num=-1,
            summary_type=0,  # 0 or 1
        )

    def _print_info(self, chosen_dur_list, part_each_dur, start_loading_time):
        print(f" finished in: {time.time()-start_loading_time:.2f}s")

        print("".center(50, "="))
        print("[dataset info]")
        print(f"total # samples: {self.total_samples}")
        print(f"dur of {list(chosen_dur_list)}")
        print(f"part of {list(part_each_dur)} are chosen")

        print("".center(50, "-"))
        print("shapes:")
        print(f"[seqC] shape: {self.seqC_all.shape}")
        print(f"[theta] shape: {self.theta_all.shape}")
        print(f"[probR] shape: {self.probR_all.shape}")

        print("".center(50, "-"))
        print("example:")
        start_loading_time = time.time()
        print(f"[seqC] e.g. {self.__getitem__(0)[0]}")
        print(f"[theta] e.g. {self.__getitem__(0)[1]}")
        print(f"[probR] e.g. {self.__getitem__(0)[2]}")
        print(f"loading time: {1000*(time.time()-start_loading_time):.2f}ms")
        print("".center(50, "="))

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        """
        get a sample from the dataset
        seqC    (15-1) # remove the first element 0
        theta   (4)
        probR   (1)
        """

        seqC_idx, theta_idx = divmod(idx, self.T)
        # faset than unravel_index indexing

        seqC = self.seqC_all[seqC_idx, 1:]
        theta = self.theta_all[theta_idx, :]
        probR = self.probR_all[seqC_idx, theta_idx]

        return seqC, theta, probR


class chR_Comb_Dataset(probR_Comb_Dataset):
    """My_Dataset:  load data into memory and finish the probR sampling with C times
    with My_HighD_Sets
    __getitem__:
        x :     (DM, S, L)
        theta:  (4,)

    loaded into memory:
        seqC_normed     of shape (M*_S, L)       - (3*S, 15)
        probR           of shape (M*_S, T, 1)    - (3*S, 500, 1)
        chR(offline)    of shape (M*_S, T, C, 1) - (3*S, 500, 100, 1)
        theta           of shape (T, 4)          - (500, 4)

    get item:
        x               of shape (14+1) - (21, 700, 15)
        theta           of shape (4)    - (4)

    """

    def __init__(
        self,
        data_dir,
        num_max_theta=500,
        num_chosen_theta=500,
        chosen_dur_list=[3, 9, 15],
        part_each_dur=[1, 1, 1],
        last_part=False,
        theta_chosen_mode="random",
        num_probR_sample=100,
        probR_sample_mode="online",  # online or offline
        print_info=True,
        config_theta=None,
    ):
        start_loading_time = time.time()
        super().__init__(
            data_dir=data_dir,
            num_max_theta=num_max_theta,
            num_chosen_theta=num_chosen_theta,
            chosen_dur_list=chosen_dur_list,
            part_each_dur=part_each_dur,
            last_part=last_part,
            theta_chosen_mode=theta_chosen_mode,
            config_theta=config_theta,
        )
        self.C = num_probR_sample
        self.MS = self.seqC_all.shape[0]
        self.seqC_len = self.seqC_all.shape[-1]
        self.probR_sample_mode = probR_sample_mode

        if self.probR_sample_mode == "offline":
            print(f"\n('offline') Sampling C={self.C} times from probR ... ", end="")

            self.probR_all = self.probR_all.to(
                "cuda" if torch.cuda.is_available() else "cpu"
            ).repeat_interleave(self.C, dim=-1)
            # probR_all (MS, T, C)

            self.chR_all = (
                torch.bernoulli(self.probR_all).unsqueeze(-1).to("cpu").contiguous()
            )
            # chR_all (MS, T, C, 1)

            del self.probR_all
            clean_cache()

        self.total_samples = self.T * self.C * self.MS
        if print_info:
            self._print_info(chosen_dur_list, part_each_dur, start_loading_time)

    def _print_info(self, chosen_dur_list, part_each_dur, start_loading_time):
        print(f" in: {time.time()-start_loading_time:.2f}s")

        print("".center(50, "="))
        print("[dataset info]")
        print(f"total # samples: {self.total_samples}")
        print(f"dur of {list(chosen_dur_list)}")
        print(f"part of {list(part_each_dur)} are chosen")

        print("".center(50, "-"))
        print("shapes:")
        print(f"[seqC] shape: {self.seqC_all.shape}")
        if self.probR_sample_mode == "offline":
            print(f"[chR] shape: {self.chR_all.shape}")
        else:
            print(f"[probR] shape: {self.probR_all.shape}")
        print(f"[theta] shape: {self.theta_all.shape}")

        print("".center(50, "-"))
        print("example:")
        start_loading_time = time.time()
        print(f"[x] e.g. {self.__getitem__(0)[0]}")
        print(f"[theta] e.g. {self.__getitem__(0)[1]}")
        print(f"loading time: {1000*(time.time()-start_loading_time):.2f}ms")

        print("".center(50, "="))

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        idx_seqC, idx_T, idx_C = unravel_index(idx, shape=(self.MS, self.T, self.C))

        if self.probR_sample_mode == "offline":
            # slice the sampled choices
            seqC = self.seqC_all[idx_seqC, 1:]
            chR = self.chR_all[idx_seqC, idx_T, idx_C]

            x = torch.cat([seqC, chR], dim=-1)
            theta = self.theta_all[idx_T, :]

        else:
            # online generate choices
            seqC = self.seqC_all[idx_seqC, 1:]
            probR = self.probR_all[idx_seqC, idx_T]
            chR = torch.bernoulli(probR)

            x = torch.cat([seqC, chR], dim=-1)
            theta = self.theta_all[idx_T, :]

        return x, theta


if __name__ == "__main__":
    setup_seed(100)
    Dataset = probR_Comb_Dataset(
        data_dir="/home/ubuntu/tmp/NSC/data/dataset-comb",
        num_chosen_theta=50,
        chosen_dur_list=[3, 9],
        part_each_dur=[1, 1],
        num_max_theta=50,
        theta_chosen_mode="random",
        print_info=True,
    )
    seqC, theta, probR = Dataset[0]

    setup_seed(100)
    Dataset = probR_Comb_Dataset(
        data_dir="/home/ubuntu/tmp/NSC/data/dataset-comb",
        num_chosen_theta=500,
        chosen_dur_list=[3, 5, 7, 9],
        part_each_dur=[1, 0.5, 0.8, 0.2],
        num_max_theta=500,
        theta_chosen_mode="random",
        print_info=True,
    )
    seqC, theta, probR = Dataset[0]

    setup_seed(100)
    Dataset = chR_Comb_Dataset(
        data_dir="/home/ubuntu/tmp/NSC/data/dataset-comb",
        num_chosen_theta=50,
        chosen_dur_list=[3, 5, 7, 9],
        part_each_dur=[1, 1, 0.1, 0.1],
        num_max_theta=50,
        theta_chosen_mode="random",
        num_probR_sample=100,
        probR_sample_mode="offline",
        print_info=True,
    )
    x, theta = Dataset[0]

    setup_seed(100)
    Dataset = chR_Comb_Dataset(
        data_dir="/home/ubuntu/tmp/NSC/data/dataset-comb",
        num_chosen_theta=50,
        chosen_dur_list=[3, 9],
        part_each_dur=[1, 0.2],
        num_max_theta=500,
        theta_chosen_mode="random",
        num_probR_sample=100,
        probR_sample_mode="online",
        print_info=True,
    )
    x, theta = Dataset[0]
