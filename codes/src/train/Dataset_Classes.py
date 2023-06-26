import time
import h5py
import torch
import numpy as np
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import gc

# from dataset.data_process import process_x_seqC_part
import sys

sys.path.append("./src")
sys.path.append("../../src")
from dataset.data_process import process_x_seqC_part


def generate_permutations(N, K):
    """
    Generate random permutations.

    Args:
        N (int): The number of permutations to generate.
        K (int): The length of each permutation.

    Returns:
        torch.Tensor: A tensor of shape (N, K) containing random permutations.

    """
    return torch.rand(N, K).argsort(dim=-1)


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


# def apply_advanced_indexing_along_dim1(tensor, indices):
#     idx0 = torch.arange(tensor.size(0))[:, None].expand_as(indices)
#     idx2 = torch.arange(tensor.size(2))[None, :].expand_as(indices)
#     return tensor[idx0, indices, idx2]


def apply_advanced_indexing_along_dim1(tensor, indices):
    idx0 = torch.arange(tensor.size(0))[:, None].expand(indices.size(0), indices.size(1))
    return tensor[idx0, indices]


def get_L_seqC(seqC_process, summary_type):
    if seqC_process == "norm":
        # seqC_shape = f[chosen_set_names[0]]['seqC_normed'].shape[1]
        L = 15
    elif seqC_process == "summary":
        if summary_type == 0:
            # seqC_shape = f[chosen_set_names[0]]['seqC_summary_0'].shape[1]
            L = 11
        elif summary_type == 1:
            # seqC_shape = f[chosen_set_names[0]]['seqC_summary_1'].shape[1]
            L = 8
    return L


def choose_theta(num_chosen_theta_each_set, max_theta_in_a_set, theta_chosen_mode):
    """choose theta and return theta idx from the set of theta in the dataset"""

    assert (
        num_chosen_theta_each_set <= max_theta_in_a_set
    ), f"num_chosen_theta_each_set={num_chosen_theta_each_set} > max_theta_in_a_set={max_theta_in_a_set}"

    # choose randomly num_chosen_theta_each_set from all the theta in the set
    if theta_chosen_mode == "random":
        theta_idx = np.random.choice(max_theta_in_a_set, num_chosen_theta_each_set, replace=False)
        return np.sort(theta_idx), len(theta_idx)

    # choose first 80% as training set from num_chosen_theta_each_set
    elif theta_chosen_mode.startswith("first"):  # first_80
        percentage = eval(theta_chosen_mode[-2:]) / 100
        theta_idx = np.arange(int(num_chosen_theta_each_set * percentage))
        return theta_idx, len(theta_idx)

    # choose last 20% as validation set from num_chosen_theta_each_set
    elif theta_chosen_mode.startswith("last"):  # last_20
        percentage = 1 - eval(theta_chosen_mode[-2:]) / 100
        theta_idx = np.arange(int(num_chosen_theta_each_set * percentage), num_chosen_theta_each_set)
        return theta_idx, len(theta_idx)

    else:
        raise ValueError(f"Invalid theta_chosen_mode: {theta_chosen_mode}")


class probR_HighD_Sets(Dataset):
    def __init__(
        self,
        data_path,
        chosen_set_names,
        num_chosen_theta_each_set,
        chosen_dur=[3, 9, 15],
        crop_dur=False,
        max_theta_in_a_set=500,
        theta_chosen_mode="random",
        seqC_process="norm",
        summary_type="0",
    ):
        super().__init__()

        """Loading the high-dimensional sets into memory
        seqC            of shape set_0:(D, M, S, 15)  - (7, 3, 700, 15)
        seqC_normed     of shape set_0:(DMS, 15)      - (14700, 15) #TODO add this part into the dataset (currently removed)
        seqC_summary_0  of shape set_0:(DMS, 15)      - (14700, 11)
        seqC_summary_1  of shape set_0:(DMS, 15)      - (14700, 8)
        
        probR           of shape set_0:(D, M, S, T)   - (7, 3, 700, 5000)
        theta           of shape set_0:(T, 4)         - (5000, 4)
        
        loaded into memory:
        seqC_normed     of shape (num_sets, DM, S, L) - (7, 21, 700, 15)
        theta           of shape (num_sets, T, 4)     - (7, 5000, 4)
        probR           of shape (num_sets, DM, S, T) - (7, 21, 700, 5000) #TODO fixed probR sampling, in init function
        
        get item:
        seqC_normed     of shape (DM, S, L)           - (21, 700, 15)
        theta           of shape (4)                  - (4)
        probR           of shape (DM, S, 1)           - (21, 700, 1)
        
        
        theta_chosen_mode:
            'random from all'   - randomly choose 'num_chosen_theta_each_set' theta from all the theta in the set
            'first 80per from chosen' - choose first 80% from 'num_chosen_theta_each_set' (normally as training set)
            'last 20per from chosen' - choose first 20% from 'num_chosen_theta_each_set' (normally as validation set)
        """
        # set seed
        # np.random.seed(config.seed)
        # torch.manual_seed(config.seed)
        # torch.cuda.manual_seed(config.seed)

        dur_list = [3, 5, 7, 9, 11, 13, 15]
        print(
            f"Loading {len(chosen_set_names)} dataset into memory... \n{chosen_set_names} ...",
            end=" ",
        )

        start_loading_time = time.time()
        self.num_chosen_theta_each_set = num_chosen_theta_each_set
        self.chosen_set_names = chosen_set_names

        f = h5py.File(data_path, "r", libver="latest", swmr=True)

        # get the shape of the data
        L_seqC = get_L_seqC(seqC_process, summary_type)
        D, M, S, DMS = (
            *f[chosen_set_names[0]]["seqC"].shape[:-1],
            np.prod(f[chosen_set_names[0]]["seqC"].shape[:-1]),
        )
        self.D, self.M, self.S = D, M, S

        # define the final shape of the data
        chosen_theta_idx, num_chosen_theta_each_set = choose_theta(
            num_chosen_theta_each_set, max_theta_in_a_set, theta_chosen_mode
        )
        self.total_samples = num_chosen_theta_each_set * len(chosen_set_names)
        self.T = num_chosen_theta_each_set

        chosen_dur_idx = [
            dur_list.index(dur) for dur in chosen_dur
        ]  # mapping from [3, 5, 7, 9, 11, 13, 15] to [0, 1, 2, 3, 4, 5, 6]
        chosen_D = len(chosen_dur) if crop_dur else D
        self.seqC_all = np.empty((len(chosen_set_names), chosen_D, M, S, L_seqC))  # (n_set, D, M, S, L)
        self.theta_all = np.empty((len(chosen_set_names), num_chosen_theta_each_set, 4))  # (n_set, T, 4)
        self.probR_all = np.empty(
            (len(chosen_set_names), chosen_D, M, S, num_chosen_theta_each_set, 1)
        )  # (n_set, D, M, S, T, 1)

        counter = 0
        for set_idx, set_name in enumerate(chosen_set_names):
            if counter % 2 == 0:
                print(counter, end=" ")

            seqC_data = self._get_seqC_data(
                crop_dur, f, seqC_process, summary_type, chosen_dur_idx, set_name
            )  # (D, M, S, L)
            probR_data = (
                f[set_name]["probR"][chosen_dur_idx, :, :, :][:, :, :, chosen_theta_idx]
                if crop_dur
                else f[set_name]["probR"][:, :, :, chosen_theta_idx]
            )  # (D, M, S, T)

            self.theta_all[set_idx] = f[set_name]["theta"][:][chosen_theta_idx, :]  # (T, 4)
            self.seqC_all[set_idx] = seqC_data
            self.probR_all[set_idx] = probR_data
            del seqC_data, probR_data
            counter += 1

        if not crop_dur:
            self.seqC_all[:, chosen_dur_idx, :, :, :] = 0  # (n_set, D, M, S, L)
            self.probR_all[:, chosen_dur_idx, :, :, :] = 0  # (n_set, D, M, S, T, 1)

        self._print_info(chosen_dur, crop_dur, start_loading_time)

        f.close()

        # convert to tensor and reshape
        self.seqC_all = (
            torch.from_numpy(self.seqC_all)
            .reshape(len(chosen_set_names), chosen_D * M, S, L_seqC)
            .to(torch.float32)
            .contiguous()
        )  # (n_set, DM, S, L)
        self.probR_all = (
            torch.from_numpy(self.probR_all)
            .reshape(len(chosen_set_names), chosen_D * M, S, num_chosen_theta_each_set)
            .to(torch.float32)
            .contiguous()
        )  # (n_set, DM, S, T)
        self.theta_all = torch.from_numpy(self.theta_all).to(torch.float32).contiguous()  # (n_set, T, 4)

    def _get_seqC_data(self, crop_dur, f, seqC_process, summary_type, chosen_dur_idx, set_name):
        seqC = (
            f[set_name]["seqC"][chosen_dur_idx, :, :, :] if crop_dur else f[set_name]["seqC"][:]
        )  # (D, M, S, L)
        return process_x_seqC_part(
            seqC=seqC,
            seqC_process=seqC_process,  # 'norm' or 'summary'
            nan2num=-1,
            summary_type=summary_type,  # 0 or 1
        )

    def _print_info(self, chosen_dur, crop_dur, start_loading_time):
        print(f" finished in: {time.time()-start_loading_time:.2f}s")
        if crop_dur:
            print(f"dur of {list(chosen_dur)} are chosen, others are [removed] ")
        else:
            print(
                f"dur of {list(chosen_dur)} are chosen, others are [set to 0] (crop_dur is suggested to be set as True)"
            )
        print(f"[seqC] shape: {self.seqC_all.shape}")
        print(f"[theta] shape: {self.theta_all.shape}")
        print(f"[probR] shape: {self.probR_all.shape}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        """
        get a sample from the dataset
        seqC    (DM, S, L)
        theta   (4)
        probR   (DM, S, 1)
        """

        set_idx, theta_idx = divmod(idx, self.num_chosen_theta_each_set)  # faset than unravel_index indexing

        # set_idx, theta_idx = self.set_idxs[idx], self.theta_idxs[idx]

        seqC = self.seqC_all[set_idx]
        theta = self.theta_all[set_idx, theta_idx]
        probR = self.probR_all[set_idx, :, :, theta_idx][:, :, np.newaxis]

        return seqC, theta, probR


class probR_2D_Sets(probR_HighD_Sets):
    """
    get item: highD -> 2D
        seqC    (DM, S, L)
        theta   (4)
        probR   (DM, S, 1)
    ->
        seqC    (DMS, L)
        theta   (4)
        probR   (DMS, 1)
    """

    def __init__(
        self,
        data_path,
        chosen_set_names,
        num_chosen_theta_each_set,
        chosen_dur=[3, 9, 15],
        crop_dur=False,
        max_theta_in_a_set=500,
        theta_chosen_mode="random",
        seqC_process="norm",
        summary_type="0",
    ):
        super().__init__(
            data_path=data_path,
            chosen_set_names=chosen_set_names,
            num_chosen_theta_each_set=num_chosen_theta_each_set,
            chosen_dur=chosen_dur,
            crop_dur=crop_dur,
            max_theta_in_a_set=max_theta_in_a_set,
            theta_chosen_mode=theta_chosen_mode,
            seqC_process=seqC_process,
            summary_type=summary_type,
        )

        # seqC_all  (n_set, DM, S, L)
        # probR_all (n_set, DM, S, T)
        # theta_all (n_set, T, 4)

        num_sets = len(chosen_set_names)
        DMS, T = self.D * self.M * self.S, self.T

        self.seqC_all = self.seqC_all.reshape(num_sets, DMS, -1)  # (n_set, DMS, L)
        self.probR_all = self.probR_all.reshape(num_sets, DMS, T)  # (n_set, DMS, T)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        set_idx, theta_idx = divmod(idx, self.num_chosen_theta_each_set)  # faset than unravel_index indexing
        # set_idx, theta_idx = self.set_idxs[idx], self.theta_idxs[idx]

        seqC = self.seqC_all[set_idx]
        theta = self.theta_all[set_idx, theta_idx]
        probR = self.probR_all[set_idx, :, theta_idx][:, np.newaxis]

        return seqC, theta, probR


class chR_HighD_Dataset(probR_HighD_Sets):
    """My_Dataset:  load data into memory and finish the probR sampling with C times
    with My_HighD_Sets
    __getitem__:
        x :     (DM, S, L)
        theta:  (4,)

    loaded into memory:
        seqC_normed     of shape (num_sets, DM, S, L) - (7, 21, 700, 15)
        theta           of shape (num_sets, T, 4)     - (7, 5000, 4)
        probR           of shape (num_sets, DM, S, T) - (7, 21, 700, 5000) #TODO fixed probR sampling, in init function

    get item:
        seqC_normed     of shape (DM, S, L)           - (21, 700, 15)
        theta           of shape (4)                  - (4)
        probR           of shape (DM, S, 1)           - (21, 700, 1)

    all data:
        seqC_all:          (num_chosen_sets, DM, S, 15 or L_x)
        theta_all:         (num_chosen_sets, T, 4)
        probR_all:         (num_chosen_sets, DM, S, T)

    """

    def __init__(
        self,
        data_path,
        chosen_set_names,
        num_chosen_theta_each_set,
        chosen_dur=[3, 9, 15],
        crop_dur=False,
        max_theta_in_a_set=500,
        theta_chosen_mode="random",
        seqC_process="norm",
        summary_type="0",
        permutation_mode="online",  # online or offline
        num_probR_sample=100,
    ):
        super().__init__(
            data_path=data_path,
            chosen_set_names=chosen_set_names,
            num_chosen_theta_each_set=num_chosen_theta_each_set,
            chosen_dur=chosen_dur,
            crop_dur=crop_dur,
            max_theta_in_a_set=max_theta_in_a_set,
            theta_chosen_mode=theta_chosen_mode,
            seqC_process=seqC_process,
            summary_type=summary_type,
        )
        # self.seqC_all = seqC_all  # shape (num_chosen_sets, DM, S, 15 or L_x)
        # self.theta_all = theta_all # shape (num_chosen_sets, T, 4)
        # self.probR_all = probR_all # shape (num_chosen_sets, DM, S, T)

        self.C = num_probR_sample
        print(
            f"==>> Further Sampling {self.C} times from probR (given 'in_dataset' process setting) ... ",
            end="",
        )
        time_start = time.time()
        self.DM, self.S, self.T = (
            self.seqC_all.shape[1],
            self.seqC_all.shape[2],
            self.theta_all.shape[1],
        )
        # probR_all (num_chosen_sets, DM, S, T, C)
        self.probR_all = (
            self.probR_all.to("cuda" if torch.cuda.is_available() else "cpu")
            .unsqueeze(-1)
            .repeat_interleave(self.C, dim=-1)
        )
        # chR_all (num_chosen_sets, DM, S, T, C, 1)
        self.chR_all = torch.bernoulli(self.probR_all).unsqueeze(-1).to("cpu").contiguous()
        print(f"in {(time.time()-time_start)/60:.2f}min")
        del self.probR_all
        torch.cuda.empty_cache()

        self.total_samples = len(chosen_set_names) * self.T * self.C
        info = f"sampled chR shape {self.chR_all.shape} MEM size {self.chR_all.element_size()*self.chR_all.nelement()/1024**3:.2f}GB, Total samples: {self.total_samples} "
        print(info)
        # self.permutations = generate_permutations(self.C*self.T*len(chosen_set_names)*self.DM, self.S).contiguous() # (C*T*Set*DM, S) #TODO use this way to shuffle
        # TODO: maybe wrong, current is (TC, s), different head share the same shuffling method
        if permutation_mode == "offline":
            self.permutations = generate_permutations(self.total_samples, self.S).contiguous()

        self.seqC_all = self.seqC_all.contiguous()  # (num_chosen_sets, DM, S, 15)
        self.theta_all = self.theta_all  # (num_chosen_sets, T, 4)
        self.chR_all = self.chR_all  # (num_chosen_sets, DM, S, T, C, 1)

        # indices = torch.arange(self.total_samples)
        # self.set_idxs, self.theta_idxs, self.probR_sample_idxs = unravel_index(indices, (len(self.chosen_set_names), self.T, self.C))
        indices = torch.arange(self.total_samples)
        self.set_idxs, self.T_idxs, self.C_idxs = unravel_index(
            indices, (len(self.chosen_set_names), self.T, self.C)
        )

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Calculate set index and theta index within the set
        set_idx, T_idx, C_idx = self.set_idxs[idx], self.T_idxs[idx], self.C_idxs[idx]
        # set_idx, theta_idx, probR_sample_idx = self.set_idxs[idx], self.theta_idxs[idx], self.probR_sample_idxs[idx]

        x = torch.empty((self.DM, self.S, self.seqC_all.shape[-1] + 1), dtype=torch.float32)
        # TODO more complex shuffling methods? currently along S axis
        x[:, :, : self.seqC_all.shape[-1]] = self.seqC_all[set_idx]  # (DM, S, 15)
        x[:, :, self.seqC_all.shape[-1] :] = self.chR_all[set_idx, :, :, T_idx, C_idx, :]  # (DM, S, 1)

        # shuffle along S
        if self.permutation_mode == "offline":
            x = x[:, self.permutations[idx], :]  # (DM, S, 15+1)
        elif self.permutation_mode == "online":
            x = x[:, torch.randperm(self.S), :]  # (DM, S, 15+1)

        theta = self.theta_all[set_idx, T_idx]  # (4,)

        return x, theta


class chR_2D_Dataset(chR_HighD_Dataset):
    """
    output 2D dataset of shape
    x: (DMS, 15 or L_x)
    theta: (4)
    """

    def __init__(
        self,
        data_path,
        chosen_set_names,
        num_chosen_theta_each_set,
        chosen_dur=[3, 9, 15],
        crop_dur=False,
        max_theta_in_a_set=500,
        theta_chosen_mode="random",
        seqC_process="norm",
        summary_type="0",
        permutation_mode="online",
        num_probR_sample=100,
    ):
        super().__init__(
            data_path=data_path,
            chosen_set_names=chosen_set_names,
            num_chosen_theta_each_set=num_chosen_theta_each_set,
            chosen_dur=chosen_dur,
            crop_dur=crop_dur,
            max_theta_in_a_set=max_theta_in_a_set,
            theta_chosen_mode=theta_chosen_mode,
            seqC_process=seqC_process,
            summary_type=summary_type,
            permutation_mode=permutation_mode,
            num_probR_sample=num_probR_sample,
        )
        DM, S, C = self.DM, self.S, self.C
        self.DMS = DM * S

        if permutation_mode == "offline":  # (nSets*TC, DMS)
            self.permutations = generate_permutations(self.total_samples, DM * S).contiguous()

        self.permutation_mode = permutation_mode

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        set_idx, T_idx, C_idx = self.set_idxs[idx], self.T_idxs[idx], self.C_idxs[idx]

        seqC = self.seqC_all[set_idx, :, :, :]  # (DM, S, 15)
        theta = self.theta_all[set_idx, T_idx, :]  # (4,)
        choice = self.chR_all[set_idx, :, :, T_idx, C_idx, :]  # (DM, S, 1)
        x = torch.cat((seqC, choice), dim=-1).reshape(self.DM * self.S, -1)  # (DMS, 16)

        if self.permutation_mode == "offline":
            x = x[self.permutations[idx], :]  # (DMS, 16)
        elif self.permutation_mode == "online":
            x = x[torch.randperm(self.DMS), :]  # (DMS, 16)
        else:
            raise NotImplementedError

        return x, theta


class Data_Prefetcher:
    def __init__(self, loader, prefetch_factor=3):
        # torch.manual_seed(config.seed)
        # torch.cuda.manual_seed(config.seed)

        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.prefetch_factor = prefetch_factor
        self.prefetched_data = []
        self.preload()

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            for _ in range(self.prefetch_factor):
                input, target = next(self.loader)
                with torch.cuda.stream(self.stream):
                    input = input.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                self.prefetched_data.append((input, target))
                # print(f'prefetcher preloaded {len(self.prefetched_data)}')

        except StopIteration:
            self.prefetched_data.append((None, None))

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if len(self.prefetched_data) == 0:
            return None, None

        input_, target = self.prefetched_data.pop(0)
        # print(f'prefetcher next called: prefetcher {len(self.prefetched_data)}')
        self.preload()  # start preloading next batches
        return input_, target

        # torch.cuda.current_stream().wait_stream(self.stream)
        # input_ = self.next_input
        # target = self.next_target
        # self.preload()
        # return input_, target


def collate_fn_vec_high_dim(batch, config, shuffling_method=0, debug=False):
    """
    batch: [
            (seqC, theta, probR),
            (seqC, theta, probR),
            ...
            (seqC, theta, probR),
            ]
            seqC: (DM, S, L), theta: (4,), probR: (DM, S, 1)

    """

    C = config["dataset"]["num_probR_sample"]
    B = len(batch)

    if debug:
        start_time_0 = time.time()

    # # Preallocate tensors
    # seqC_batch = torch.empty((B, *batch[0][0].shape))
    # theta_batch = torch.empty((B, *batch[0][1].shape))
    # probR_batch = torch.empty((B, *batch[0][2].shape))

    # # Fill tensors with data from the batch
    # for i, (seqC, theta, probR) in enumerate(batch):
    #     seqC_batch[i] = seqC
    #     theta_batch[i] = theta
    #     probR_batch[i] = probR

    # seqC, theta, probR = zip(*batch)

    # seqC_batch = torch.stack(seqC)
    # theta_batch = torch.stack(theta)
    # probR_batch = torch.stack(probR)

    seqC_batch, theta_batch, probR_batch = map(torch.stack, zip(*batch))

    if debug:
        print(f"collate_fn_vec_high_dim: dataloading {(time.time() - start_time_0)*1000:.2f} ms")

    # del batch, seqC, theta, probR

    # Repeat seqC and theta C times along a new dimension
    seqC_batch = seqC_batch.repeat_interleave(
        C, dim=0
    )  # (C*B, DM, S, 15) first C samples are the same, from the first batch
    theta_batch = theta_batch.repeat_interleave(C, dim=0)  # (C*B, 4)

    # Repeat probR C times along a new dimension and sample from Bernoulli distribution
    probR_batch = probR_batch.repeat_interleave(C, dim=0)  # (C*B, DM, S, 1)

    # bernouli sampling choice, and concatenate x_seqC and x_choice
    x_batch = torch.cat([seqC_batch, probR_batch.bernoulli_()], dim=-1)  # (C*B, DM, S, 16)
    del probR_batch, seqC_batch
    gc.collect()

    if debug:
        print(f"\ncollate_fn_vec: get x_batch {(time.time() - start_time_0)*1000:.2f} ms")
        start_time = time.time()

    # Shuffle x along the 3rd axis S, using advanced indexing
    BC, DM, S, _ = x_batch.shape
    permutations = torch.stack(
        [torch.stack([torch.randperm(S) for _ in range(DM)]) for _ in range(BC)]
    )  # (BC, DM, S)

    # for i in range(BC):
    #     for j in range(DM):
    #         x_batch[i, j] = x_batch[i, j, permutations[i, j]]
    # Shuffle the batched dataset

    indices = torch.randperm(BC)
    BC_range = indices[:, None, None]
    DM_range = torch.arange(DM)[None, :, None]

    # if debug:
    #     print(f"collate_fn_vec: shuffle x_batch {(time.time() - start_time)*1000:.2f} ms")
    #     start_time = time.time()

    return (
        x_batch[BC_range, DM_range, permutations],
        theta_batch[indices],
    )  # TODO check the output shape and shuffling result and logic

    # return x_batch[indices[:, None], :, permutations, :], theta_batch[indices]


def collate_fn_vec(batch, config, shuffling_method=0, debug=False):
    """
    batch: [
            (seqC, theta, probR),
            (seqC, theta, probR),
            ...
            (seqC, theta, probR),
            ]
            seqC: (D*M*S, 15), theta: (4,), probR: (D*M*S, 1)

            shuffling_method: 0: complex shuffle - expand x from (B, D*M*S, 16) to (B*C, D*M*S, 16) then shuffle along the 2nd axis, then shuffle the batch BC
                              1: simple shuffle - shuffle x (B, D*M*S, 16) along the 2nd axis, then expand x to (B*C, D*M*S, 16)
    """

    C = config["dataset"]["num_probR_sample"]
    B = len(batch)

    if debug:
        start_time_0 = time.time()

    # Preallocate tensors
    seqC_batch = torch.empty((B, *batch[0][0].shape))
    theta_batch = torch.empty((B, *batch[0][1].shape))
    probR_batch = torch.empty((B, *batch[0][2].shape))

    # Fill tensors with data from the batch
    for i, (seqC, theta, probR) in enumerate(batch):
        seqC_batch[i] = seqC
        theta_batch[i] = theta
        probR_batch[i] = probR

    # seqC, theta, probR = zip(*batch)

    # seqC_batch = torch.stack(seqC)
    # theta_batch = torch.stack(theta)
    # probR_batch = torch.stack(probR)

    if debug:
        print(f"collate_fn_vec: dataloading {(time.time() - start_time_0)*1000:.2f} ms")

    del batch, seqC, theta, probR

    if shuffling_method == 0:
        # Repeat seqC and theta C times along a new dimension
        seqC_batch = seqC_batch.repeat_interleave(
            C, dim=0
        )  # (C*B, D*M*S, 15) first C samples are the same, from the first batch
        theta_batch = theta_batch.repeat_interleave(C, dim=0)  # (C*B, 4)

        # Repeat probR C times along a new dimension and sample from Bernoulli distribution
        probR_batch = probR_batch.repeat_interleave(C, dim=0)  # (C*B, D*M*S, 1)
        # probR_batch = torch.bernoulli(probR_batch)  # (C*B, D*M*S, 1)
        # probR_batch.bernoulli_() # (C*B, D*M*S, 1)

        # Concatenate x_seqC and x_choice
        x_batch = torch.cat([seqC_batch, probR_batch.bernoulli_()], dim=-1)  # (C*B, D*M*S, 16)
        del probR_batch, seqC_batch
        gc.collect()

        if debug:
            print(f"\ncollate_fn_vec: get x_batch {(time.time() - start_time_0)*1000:.2f} ms")
            start_time = time.time()

        # Shuffle x along the 2nd axis
        # x_batch = torch.stack([x_batch[i,:,:][torch.randperm(x_batch.shape[1]),:] for i in range(x_batch.shape[0])])
        DMS = x_batch.shape[1]
        # x_batch_shuffled = torch.empty_like(x_batch)

        # permutations = generate_permutations(B*C, DMS)
        # permutations = list(map(lambda _: torch.randperm(DMS), range(B*C)))

        # permutations = [torch.randperm(DMS) for _ in range(B*C)]
        permutations = torch.stack([torch.randperm(DMS) for _ in range(B * C)])
        # permutations = torch.rand(B*C, DMS).argsort(dim=-1)

        # if debug:
        #     print(f"\ncollate_fn_vec: generate permutations {(time.time() - start_time)*1000:.2f} ms")
        #     start_time = time.time()
        # start_time = time.time()

        # for i in range(B*C):
        # x_batch_shuffled[i] = x_batch[i][permutations[i]]
        # x_batch_shuffled[i] = x_batch[i][torch.randperm(DMS)]

        # gathering method
        # indices = torch.argsort(torch.rand(*x_batch.shape[:2]), dim=1)
        # x_batch_shuffled = torch.gather(x_batch, dim=1, index=indices.unsqueeze(-1).repeat(1, 1, x_batch.shape[-1]))

        # del x_batch

        if debug:
            print(f"collate_fn_vec: shuffle x_batch {(time.time() - start_time)*1000:.2f} ms")
            start_time = time.time()

            # permutations = torch.stack([torch.randperm(DMS) for _ in range(B*C)])
            # x_batch_shuffled = x_batch[torch.arange(B * C)[:, None], permutations]
            # x_batch_shuffled_2 = x_batch[torch.arange(B * C)[:, None], permutations]
            # print(f"collate_fn_vec: shuffle x_batch_2 {(time.time() - start_time)*1000:.2f} ms")
            # # print(f'same? {torch.all(torch.eq(x_batch_shuffled, x_batch_shuffled_2))}')
            # start_time = time.time()

        # Shuffle the batched dataset
        # indices             = torch.randperm(x_batch_shuffled.shape[0])
        indices = torch.randperm(x_batch.shape[0])
        # x_batch_shuffled    = x_batch_shuffled[indices]
        # theta_batch         = theta_batch[indices]

        # if debug:
        #     print(f"collate_fn_vec: finish shuffle {(time.time() - start_time)*1000:.2f} ms")
        #     print(f"collate_fn_vec: -- finish computation {(time.time() - start_time_0)*1000:.2f} ms")

        # return x_batch_shuffled[indices], theta_batch[indices]
        # shuffle along the 1st axis individually and then shuffle the batch
        # return x_batch[torch.arange(B * C)[:, None], permutations][indices], theta_batch[indices]
        return x_batch[indices[:, None], permutations], theta_batch[indices]

    elif shuffling_method == 1:
        # shuffle seqC_batch and theta_batch along the 2nd axis
        DMS = seqC_batch.shape[1]
        for i in range(B):
            indices = torch.randperm(DMS)
            seqC_batch[i] = seqC_batch[i][indices]
            probR_batch[i] = probR_batch[i][indices]

        theta_batch = theta_batch.repeat_interleave(C, dim=0)  # (C*B, 4)
        seqC_batch = seqC_batch.repeat_interleave(C, dim=0)  # (C*B, D*M*S, 15)
        probR_batch = probR_batch.repeat_interleave(C, dim=0)  # (C*B, D*M*S, 1)
        probR_batch = torch.bernoulli(probR_batch)  # (C*B, D*M*S, 1)

        x_batch = torch.cat([seqC_batch, probR_batch], dim=-1)  # (C*B, D*M*S, 16)
        del seqC_batch, probR_batch

        return x_batch, theta_batch


def collate_fn(batch, config, debug=False):
    C = config["dataset"]["num_probR_sample"]

    if debug:
        start_time_0 = time.time()

    x_batch, theta_batch = [], []

    x_batch = torch.empty(
        (
            C * len(batch),
            batch[0][0].shape[0],
            batch[0][0].shape[1] + batch[0][2].shape[1],
        )
    )
    theta_batch = torch.empty((C * len(batch), batch[0][1].shape[0]))

    for i, (seqC, theta, probR) in enumerate(batch):  # seqC: (D*M*S, 15), theta: (4,), probR: (D*M*S, 1)
        probR = probR.unsqueeze_(dim=0).repeat_interleave(C, dim=0)  # (C, D*M*S, 1)
        x_seqC = seqC.unsqueeze_(dim=0).repeat_interleave(C, dim=0)  # (C, D*M*S, 15)
        x_choice = torch.bernoulli(probR)  # (C, D*M*S, 1)

        x = torch.cat([x_seqC, x_choice], dim=-1)
        theta = theta.unsqueeze_(dim=0).repeat_interleave(C, dim=0)  # (C, 4)

        x_batch[i * C : (i + 1) * C] = x
        theta_batch[i * C : (i + 1) * C] = theta

    if debug:
        print(f"\ncollate_fn: get x_batch {(time.time() - start_time_0)*1000:.2f} ms")

    if debug:
        start_time = time.time()
    # Shuffle x along the 2nd axis
    x_batch = torch.stack([x_batch[i][torch.randperm(x_batch.shape[1])] for i in range(x_batch.shape[0])])
    if debug:
        print(f"collate_fn: shuffle x_batch {(time.time() - start_time)*1000:.2f} ms")

    if debug:
        start_time = time.time()
    # Shuffle the batched dataset
    indices = torch.randperm(x_batch.shape[0])
    x_batch = x_batch[indices]
    theta_batch = theta_batch[indices]
    if debug:
        print(f"collate_fn: finish shuffle {(time.time() - start_time)*1000:.2f} ms")
        print(f"collate_fn: -- finish computation {(time.time() - start_time_0)*1000:.2f} ms")

    return x_batch.to(torch.float32), theta_batch.to(torch.float32)


def collate_fn_probR(batch, Rchoice_method="probR_sampling", num_probR_sample=10):
    """OLD VERSION
    batch is a list of tuples, each tuple is (theta, x, prior_masks)
        original shapes:
        theta.shape = (T*C, L_theta)
        x.shape     = (T*C, DMS, L_x)
        (sequence should be shuffled when batched into the dataloader)

        e.g. C = 1  if Rchoice_method == 'probR'
        e.g. C = 10 if Rchoice_method == 'probR_sampling'
    """

    theta, x, _ = zip(*batch)

    theta = torch.stack(theta)
    x = torch.stack(x)
    _ = torch.stack(_)

    if Rchoice_method == "probR":
        # repeat theta and x for each probR sample
        theta_new = theta.repeat_interleave(num_probR_sample, dim=0)  # (T*C, L_theta)
        x_new = x.repeat_interleave(num_probR_sample, dim=0)  # (T*C, DMS, 15+1)
        _ = _.repeat_interleave(num_probR_sample, dim=0)  # (T*C, 1)
        x_seqC = x_new[:, :, :-1]  # (T*C, DMS, 15)
        x_probRs = x_new[:, :, -1].unsqueeze_(dim=2)  # (T*C, DMS, 1)

        # sample Rchoice from probR with Bernoulli
        x_Rchoice = torch.bernoulli(x_probRs)
        x = torch.cat((x_seqC, x_Rchoice), dim=2)
        theta = theta_new

    return theta, x, _


if __name__ == "__main__":
    batch_size = 20

    dataset_path = "/home/wehe/tmp/NSC/data/dataset/dataset-L0-Eset0-100sets-T500.h5"
    dataset = chR_2D_Dataset(
        data_path=dataset_path,
        chosen_set_names=["set_0", "set_1"],
        num_chosen_theta_each_set=500,
        chosen_dur=[3, 9, 15],
        crop_dur=False,
        max_theta_in_a_set=500,
        theta_chosen_mode="random",
        seqC_process="norm",
        summary_type="0",
        permutation_mode="online",
        num_probR_sample=100,
    )
    x = dataset[0][0]
    theta = dataset[0][1]
    print(f"==>> x.shape: {x.shape}")
    print(f"==>> theta.shape: {theta.shape}")

    # prepare train, val, test dataloader
    loader_kwargs = {
        "batch_size": min(
            batch_size,
            len(dataset),
        ),
        "drop_last": True,
        "shuffle": True,
        "pin_memory": True,
        "num_workers": 4,
        "prefetch_factor": 2,
        # "worker_init_fn": seed_worker,
    }
    print(f"{loader_kwargs=}")

    g = torch.Generator()
    g.manual_seed(100)

    train_dataloader = data.DataLoader(dataset, generator=g, **loader_kwargs)
    x, theta = next(iter(train_dataloader))
