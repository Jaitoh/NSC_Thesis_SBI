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


class BaseDataset(Dataset):
    def __init__(
        self,
        data_dir,
        chosen_dur_list=[3, 9, 15],
        num_max_seqC_each_dur=[8, 2000, 5000],
        last_seqC_part=False,
        num_max_theta=500,
        num_chosen_theta=500,
        theta_chosen_mode="random",
        print_info=False,
    ):
        super().__init__()

        """Loading the high-dimensional sets into memory
        ---
        original data:
        seqC            of shape:(1, M, _S, L)      - (1, 3, 3^n-1-1, 15)
        probR           of shape:(1, M, _S, T, 1)   - (1, 3, 3^n-1-1, 500, 1)
        theta           of shape:(1, M, _S, T, 4)   - (1, 3, 3^n-1-1, 500, 4)
        
        ---
        loaded into memory:
        seqC_normed     of shape (M, _S, L)          - (3, S, 15)
        probR           of shape (M, _S, T, 1)       - (3, S, 500, 1)
        theta           of shape (M, _S, T, 4)       - (3, S, 500, 4)
        
        
        theta_chosen_mode:
            'random'   - randomly choose 'num_chosen_theta' theta from all the theta in the set
            'first_80' - choose first 80% from 'num_chosen_theta' (normally as training set)
            'last_20'  - choose first 20% from 'num_chosen_theta' (normally as validation set)
        """
        # start_loading_time = time.time()
        print("start loading data into MEM ... ")
        self.num_chosen_theta = num_chosen_theta

        dataset_path = adapt_path(Path(data_dir) / f"dataset-partcomb-T500.h5")  # TODO: change
        # define the final shape of the data
        chosen_theta_idx, num_chosen_theta = choose_theta(num_chosen_theta, num_max_theta, theta_chosen_mode)
        self.T = num_chosen_theta
        self.S = 0
        self.S_each_dur = []
        f = h5py.File(dataset_path, "r")
        _, self.M, _, self.L_seqC = f[f"dur_3"]["seqC"].shape  # (1, M, S, L)
        f.close()

        for dur, num_seqC in zip(chosen_dur_list, num_max_seqC_each_dur):
            self.S += num_seqC
            self.S_each_dur.append(num_seqC)

        self.seqC_all = np.zeros((self.M, self.S, self.L_seqC))
        self.probR_all = np.zeros((self.M, self.S, self.T))
        self.theta_all = np.zeros((self.M, self.S, self.T, 4))

        self.total_samples = self.M * self.S * self.T

        S_cnt, counter = 0, 0
        f = h5py.File(dataset_path, "r")
        for dur, num_seqC in zip(chosen_dur_list, num_max_seqC_each_dur):
            print(counter, end=" ")  # if counter % 2 == 0 else None

            # load data # libver="latest",swmr=True
            seqC_data = self._get_seqC_data(f, dur, num_seqC, last_seqC_part=last_seqC_part)
            # seqC_data (M, _S, L)

            _S = seqC_data.shape[1]
            probR_data = (
                f[f"dur_{dur}"]["probR"][:, :, -_S:, chosen_theta_idx, 0][0]
                if last_seqC_part
                else f[f"dur_{dur}"]["probR"][:, :, :_S, chosen_theta_idx, 0][0]
            )
            # probR_data (M, _S, T, 1)

            theta_data = (
                f[f"dur_{dur}"]["theta"][:, :, -_S:, chosen_theta_idx, :][0]
                if last_seqC_part
                else f[f"dur_{dur}"]["theta"][:, :, :_S, chosen_theta_idx, :][0]
            )
            # theta_data (M, _S, T, 4)

            self.seqC_all[:, S_cnt : (S_cnt + _S), :] = seqC_data
            self.probR_all[:, S_cnt : (S_cnt + _S), :] = probR_data
            self.theta_all[:, S_cnt : (S_cnt + _S), :, :] = theta_data

            del seqC_data, probR_data, theta_data
            counter += 1
            S_cnt += _S
        f.close()

    def _get_seqC_data(self, f, dur, num_seqC, last_seqC_part=False):
        seqC_shape = f[f"dur_{dur}"]["seqC"].shape  # seqC: (1, M, S, L)
        S = seqC_shape[2]
        # take partial of S
        seqC = (
            f[f"dur_{dur}"]["seqC"][:, :, -num_seqC:, :][0]
            if last_seqC_part
            else f[f"dur_{dur}"]["seqC"][:, :, :num_seqC, :][0]
        )
        # seqC: (M, S, L)

        return process_x_seqC_part(
            seqC=seqC,
            seqC_process="norm",  # 'norm' or 'summary'
            nan2num=-1,
            summary_type=0,  # 0 or 1
        )


class probR_Comb_Dataset(BaseDataset):
    def __init__(
        self,
        data_dir,
        chosen_dur_list=[3, 9, 15],
        num_max_seqC_each_dur=[1, 1, 1],
        last_seqC_part=False,
        num_max_theta=500,
        num_chosen_theta=500,
        theta_chosen_mode="random",
        config_theta=None,
        print_info=False,
    ):
        """
        ---
        reshape memory data:
        seqC_normed     of shape (M*_S, L)          - (3, S, 15)
        probR           of shape (M*_S, T, 1)       - (3, S, 500, 1)
        theta           of shape (M*_S, T, 4)       - (3, S, 500, 4)

        ===
        get item:
        seqC_normed     of shape (L) - (15-1)
        probR           of shape (1) - (1)
        theta           of shape (4) - (4)

        """
        start_loading_time = time.time()

        super().__init__(
            data_dir=data_dir,
            chosen_dur_list=chosen_dur_list,
            num_max_seqC_each_dur=num_max_seqC_each_dur,
            last_seqC_part=last_seqC_part,
            num_max_theta=num_max_theta,
            num_chosen_theta=num_chosen_theta,
            theta_chosen_mode=theta_chosen_mode,
        )

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
        self.theta_all = (
            torch.from_numpy(self.theta_all)
            .reshape(self.M * self.S, self.T, -1)
            .to(torch.float32)
            .contiguous()
        )

        # self.theta_all = torch.from_numpy(self.theta_all).to(torch.float32).contiguous()
        self.theta_all = process_theta_3D(  # normalize and processing of the theta values
            self.theta_all,
            ignore_ss=config_theta.ignore_ss,
            normalize_theta=config_theta.normalize,
            unnormed_prior_min=config_theta.prior_min,
            unnormed_prior_max=config_theta.prior_max,
        )

        if print_info:
            self._print_info(chosen_dur_list, num_max_seqC_each_dur, start_loading_time)

    def _print_info(self, chosen_dur_list, num_max_seqC_each_dur, start_loading_time):
        print(f" finished in: {time.time()-start_loading_time:.2f}s")

        print("".center(50, "="))
        print("[dataset info]")
        print(f"total # samples: {self.total_samples}")
        print(f"dur of {list(chosen_dur_list)}")
        print(f"part of {list(num_max_seqC_each_dur)} are chosen")

        print("".center(50, "-"))
        print("shapes:")
        print(f"[seqC_all] shape [M*_S, 15]: {self.seqC_all.shape}")
        print(f"[theta_all] shape [M*_S, T, 4]: {self.theta_all.shape}")
        print(f"[probR_all] shape [M*_S, T, 1]: {self.probR_all.shape}")

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
        probR = self.probR_all[seqC_idx, theta_idx]
        theta = self.theta_all[seqC_idx, theta_idx, :]

        return seqC, theta, probR


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
    seqC_normed     of shape (M*_S, L)      - (3*S, 15)
    probR           of shape (M*_S, T, 1)   - (3*S, 500, 1)
    theta           of shape (M*_S, T, 4)   - (500, 4)

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
        theta = self.theta_all[seqC_idx, theta_idx, :]

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

    Dataset = probR_Comb_Dataset(
        data_dir="/home/ubuntu/tmp/NSC/data/dataset-comb",
        num_chosen_theta=50,
        chosen_dur_list=[3, 5],
        num_max_seqC_each_dur=[7, 73],
        last_seqC_part=False,
        num_max_theta=50,
        theta_chosen_mode="random",
        print_info=True,
        config_theta=config_theta,
    )
    seqC, theta, probR = Dataset[0]

    setup_seed(100)
    Dataset = probR_Comb_Dataset(
        data_dir="/home/ubuntu/tmp/NSC/data/dataset-comb",
        num_chosen_theta=500,
        chosen_dur_list=[3, 5, 7],
        num_max_seqC_each_dur=[1, 8, 229],
        last_seqC_part=True,
        num_max_theta=500,
        theta_chosen_mode="random",
        print_info=True,
        config_theta=config_theta,
    )
    seqC, theta, probR = Dataset[0]

    setup_seed(100)
    Dataset = x1pR_theta_Dataset(
        data_dir="/home/ubuntu/tmp/NSC/data/dataset-comb",
        num_chosen_theta=50,
        chosen_dur_list=[3, 5],
        num_max_seqC_each_dur=[7, 73],
        last_seqC_part=False,
        num_max_theta=50,
        theta_chosen_mode="random",
        print_info=True,
        config_theta=config_theta,
    )
