"""
generate/prepare training dataset for sbi inference: x, theta
"""
import time
import sys

sys.path.append('./src')

import os

import numpy as np
import torch

from simulator.DM_model import DM_model
from dataset.seqC_pattern_summary import seqC_pattern_summary
from dataset.seqC_generator import seqC_generator

from sbi import utils as utils
from joblib import Parallel, delayed
import itertools

import h5py
from pathlib import Path
from tqdm import tqdm



def seq_norm_repeat(seqC, nan2num=-2, n_repeat=100):
    """ fill the nan of the seqC with nan2num and repeat n_repeat times

    Args:
        seqC (np.array): input sequence of 1 dim range (-1 - 1)
        nan2num (float, optional): number to fill the nan. Defaults to -2.
        n_repeat (int, optional): number of repeat. Defaults to 100.
"""

    seqC = np.nan_to_num(seqC, nan=nan2num)
    # normalize the seqC from (nan2num, 1) to (0, 1)
    seqC = (seqC - nan2num) / (1 - nan2num)
    return np.vstack([seqC] * n_repeat)


def simulate_with_summary(args):
    seqC, params, modelName, num_LR_sample, nan2num = args
    # print(seqC, params)

    model = DM_model(params, modelName=modelName)
    seqC = seqC.reshape(1, -1)
    _, probR = model.stoch_simulation(seqC)

    seqC_hat = seqC_pattern_summary(seqC).reshape(-1)
    x_ = torch.as_tensor(np.append(seqC_hat, probR))

    return (x_, params, probR)


def prepare_training_data_from_sampled_Rchoices(
        dataset_dir: str or Path,
        dur_list: list,
        nan2num=-1,
        num_probR_sample=30,
        part_of_seqC=0.5,
        part_of_prior=0.5,
        remove_sigma2i=True,
):
    """ preparing the dataset (x, theta) pair for training
    from (seqC, theta, probR) -> (x, theta)
    x is the composition of normalized sequence C and sampled probR choice

    Args:
        dataset_dir: (string) the path to the dataset
        dur_list:   (list) the list of durations to be used
        nan2num:   (float) the value to replace nan in seqC
        num_probR_sample: (int) the number of samples to be drawn from probR
        part_of_seqC: (float) 0-1 the percentage of sequence samples e.g. 50 seqC samples, take the first 10, the given value would be 0.2
        part_of_prior: (float) 0-1 the percentage of prior samples
        remove_sigma2i: (bool) whether to remove sigma2i from prior samples

    Returns:
        x: (np.array) the input data for training
        theta: (np.array) the theta for training
    """

    f = h5py.File(dataset_dir, 'r')
    prior_sample_size = f['info_group']['num_prior_sample'][()]
    seqC_sample_size  = f['info_group']['seqC_sample_size'][()]

    seqC_group = f['seqC_group']
    theta_group = f['theta_group']
    probR_group = f['probR_group']
    single_dur_seqC_shape  = seqC_group[f'seqC_dur{3}'].shape
    single_dur_theta_shape = theta_group[f'theta_dur{3}'].shape
    seqC_MS_list = f['info_group']['seqC_MS_list'][:]
    single_dur_dataset_size = single_dur_seqC_shape[0]

    # extract partial data from the dataset
    pointer_MS = list(range(0, single_dur_dataset_size, single_dur_dataset_size // len(seqC_MS_list)))
    num_pointer_seqC_in_MS = single_dur_dataset_size // len(seqC_MS_list) // prior_sample_size
    num_pointer_seqC_in_MS_chosen = int(num_pointer_seqC_in_MS * part_of_seqC)

    pointer_seqC = [list(range(pointer, pointer + num_pointer_seqC_in_MS_chosen * prior_sample_size, prior_sample_size)) for pointer in pointer_MS]
    pointer_seqC = np.array(pointer_seqC).flatten()

    theta_part_num = int(prior_sample_size * part_of_prior)

    # for pointer in pointer_seqC:
    idx_chosen = [list(range(pointer, pointer+theta_part_num)) for pointer in pointer_seqC]
    idx_chosen = np.array(idx_chosen).flatten()

    print(f'original seqC sample size {seqC_sample_size}, taken part of size {int(seqC_sample_size*part_of_seqC)}')
    print(f'original prior sample size {prior_sample_size}, taken part of size {theta_part_num}')

    datasize = len(idx_chosen) * len(dur_list) * num_probR_sample
    seqC_shape = [datasize, single_dur_seqC_shape[1]]
    theta_shape = [datasize, single_dur_theta_shape[1]]
    RChoice_shape = [datasize, 1]

    seqC, theta, RChoice = np.empty(seqC_shape), np.empty(theta_shape), np.empty(RChoice_shape)

    print('loading data...')
    start_time = time.time()
    for i, dur in tqdm(enumerate(dur_list), total=len(dur_list)):
        seqC_part = np.repeat(seqC_group[f'seqC_dur{dur}'][idx_chosen], num_probR_sample, axis=0)
        theta_part = np.repeat(theta_group[f'theta_dur{dur}'][idx_chosen], num_probR_sample, axis=0)

        probR = probR_group[f'probR_dur{dur}'][idx_chosen]
        RChoice_part = np.array([
            np.random.choice([0, 1], size=num_probR_sample, p=[1 - prob[0], prob[0]])
            for prob in probR
        ]).reshape(-1, 1)

        seqC[i * len(idx_chosen) * num_probR_sample
             : (i + 1) * len(idx_chosen) * num_probR_sample, :] = seqC_part
        theta[i * len(idx_chosen) * num_probR_sample
              : (i + 1) * len(idx_chosen) * num_probR_sample, :] = theta_part
        RChoice[i * len(idx_chosen) * num_probR_sample
                : (i + 1) * len(idx_chosen) * num_probR_sample, :] = RChoice_part
    print(f'data loaded, time used: {time.time() - start_time:.2f}s')

    # adjust seqC -> nan2num + normalize
    seqC = np.nan_to_num(seqC, nan=nan2num)
    seqC = (seqC - nan2num) / (1 - nan2num)
    # seqC[0,:]
    x = np.concatenate([seqC, RChoice], axis=1)
    # x.shape, seqC.shape, theta.shape, RChoice.shape
    f.close()

    if remove_sigma2i:
        theta = np.delete(theta, 2, axis=1)  # bsssxxx remove sigma2i (the 3rd column)

    return torch.tensor(x, dtype=torch.float32), torch.tensor(theta, dtype=torch.float32)


if __name__ == '__main__':
    # remember to generate the cython code first
    # generate the sequence C, theta, probR dataset

    test = True

    # transfer the dataset to (x, theta) pair for training
    dataset_dir_test = Path('../data/training_datasets/training_dataset_test.hdf5')
    dur_list = [3, 5, 11]
    nan2num = -1
    num_probR_sample = 30

    x, theta = prepare_training_data_from_sampled_Rchoices(
        dataset_dir=dataset_dir_test,
        dur_list=dur_list,
        nan2num=nan2num,
        num_probR_sample=num_probR_sample,
        part_of_seqC=1,
        part_of_prior=1,
    )
