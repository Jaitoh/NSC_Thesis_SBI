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
from simulator.seqC_pattern_summary import seqC_pattern_summary
from data_generator.input_c import seqCGenerator

from sbi import utils as utils
from joblib import Parallel, delayed
import itertools

import h5py
from pathlib import Path
from tqdm import tqdm

# from tqdm import tqdm


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


def simulate(args):
    # seqC, params, modelName, num_LR_sample, nan2num = args
    seqC, params, modelName = args
    # print(seqC, params)

    model = DM_model(params=params, modelName=modelName)
    _, probR = model.simulate(np.array(seqC))
    # print(f'probR: {probR}')
    # cDist = np.random.choice([0,1], p=[1-probR, probR], size=num_LR_sample) # 0: left, 1: right

    # seqC_hat = seq_norm_repeat(seqC, nan2num=nan2num, n_repeat=num_LR_sample)

    # x_ = torch.as_tensor(np.hstack([seqC_hat, cDist.reshape(-1, 1)]))

    return (seqC, params, probR)


def simulate_with_summary(args):
    seqC, params, modelName, num_LR_sample, nan2num = args
    # print(seqC, params)

    model = DM_model(params, modelName=modelName)
    seqC = seqC.reshape(1, -1)
    _, probR = model.stoch_simulation(seqC)

    seqC_hat = seqC_pattern_summary(seqC).reshape(-1)
    x_ = torch.as_tensor(np.append(seqC_hat, probR))

    return (x_, params, probR)


def simulate_dataset(
        seqCs,
        prior,
        num_prior_sample,
        modelName='B-G-L0S-O-N-',
        use_seqC_summary=False,
        summary_length=5,
        # nan2num=-2,
        # num_LR_sample=100,
        num_workers=-1,
):
    """generate dataset for training sbi inference
    
    Args:   
        seqCs:              input sequences
        prior:              prior distribution
        num_prior_sample:   number of prior samples
        modelName:          model name
        use_seqC_summary:   whether to use summary of seqC. Defaults to False.
        summary_length:     length of summary. Defaults to 5.
        num_workers:        number of workers for parallel processing (default: -1)

    """

    params = prior.sample((num_prior_sample,)).numpy()

    simulate_func = simulate_with_summary if use_seqC_summary else simulate
    dataset_size = len(
        seqCs) * num_prior_sample  # if use_seqC_summary else len(seqCs) * num_prior_sample * num_LR_sample
    seqC = np.empty((dataset_size, summary_length if use_seqC_summary else seqCs.shape[1]))
    theta = np.empty((dataset_size, params.shape[1]))
    probR = np.empty((dataset_size, 1))

    print(f'number of simulations', len(seqCs) * num_prior_sample)

    tic = time.time()
    results = Parallel(n_jobs=num_workers, verbose=1)(
        delayed(simulate_func)((seqC, param, modelName)) \
        for seqC, param in itertools.product(seqCs, params))
    toc = time.time()
    print(f'time elapsed for simulation: {toc - tic:.2f} seconds')

    print('stacking the results')
    len_results = len(results)
    tic = time.time()

    for i, (seqC_, theta_, probR_) in enumerate(results):
        seqC[i, :] = seqC_
        theta[i, :] = theta_
        probR[i, 0] = probR_
        progress_bar(i, len_results)
    toc = time.time()
    print(f'time elapsed for storing: {toc - tic:.2f} seconds')

    # print(f'x: {x}, theta: {theta}')
    return seqC, theta, probR


def progress_bar(i, total):
    # print when the progress is 2, 4, 6, ... 100%
    if i % int((total * 0.5)) == 0:
        print(f'{i / total * 100:.2f}%', end='\r')


def simulate_and_store(
        save_data_dir='../data/training_datasets/training_dataset.hdf5',

        seqC_MS_list=None,
        seqC_dur_max=14,  # 2, 4, 6, 8, 10, 12, 14 -> 7
        seqC_sample_size=700,

        prior_min=None,
        prior_max=None,
        num_prior_sample=500,

        model_name='B-G-L0S-O-N-',
        # nan2num         = -2,
        # num_LR_sample   = 10,

        test=False,
):

    """
    pre-generate training dataset for sbi inference - seqC, params, probR

    Args:
        save_data_dir:      directory to save the dataset
        seqC_MS_list:       list of seqC_MS to generate
        seqC_dur_max:       maximum duration of seqC
        seqC_sample_size:   number of seqC samples
        prior_min:          minimum values of prior distribution
        prior_max:          maximum values of prior distribution
        num_prior_sample:   number of prior samples
        model_name:         model name with the params setting information
        test:               whether to test the code. Defaults to False.

    Returns:
        -> save the dataset to the save_data_dir
    """

    # set default values
    seqC_MS_list = [0.2, 0.4, 0.8] if seqC_MS_list is None else seqC_MS_list
    prior_min = [-3.7, -36, 0, -34, 5] if prior_min is None else prior_min
    prior_max = [2.5, 71, 0, 18, 7] if prior_max is None else prior_max

    # test if the output folder exists
    with h5py.File(save_data_dir, 'w') as f:
        f.create_dataset('test', data='test')
    print(f'folder {save_data_dir} exists')

    # generate prior distribution
    prior = utils.torchutils.BoxUniform(
        low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
    )
    # num_prior_sample = int(10 ** (len(prior_min) - 1))  # 10000 in this case
    num_prior_sample = 10 if test else num_prior_sample
    print(f'prior sample size', num_prior_sample)

    # generate seqC input sequence
    seqC_sample_size = 10 if test else seqC_sample_size
    dur_list = np.arange(2, seqC_dur_max + 1, 2)

    f = h5py.File(save_data_dir, 'w')
    seqC_group = f.create_group('/seqC_group')
    theta_group = f.create_group('/theta_group')
    probR_group = f.create_group('/probR_group')

    info_group = f.create_group('/info_group')
    info_group.create_dataset("seqC_MS_list", data=seqC_MS_list)
    info_group.create_dataset("seqC_dur_max", data=seqC_dur_max)
    info_group.create_dataset("seqC_sample_size", data=seqC_sample_size)
    info_group.create_dataset("dur_list", data=dur_list)
    info_group.create_dataset("num_prior_sample", data=num_prior_sample)
    info_group.create_dataset("prior_min", data=prior_min)
    info_group.create_dataset("prior_max", data=prior_max)
    info_group.create_dataset("model_name", data=model_name)
    # info_group.create_dataset("nan2num", data = nan2num)
    # info_group.create_dataset("num_LR_sample", data = num_LR_sample)

    for dur in dur_list:
        print(f'\nprocessing duration {dur}...')

        seqC_gen = seqCGenerator()
        seqCs = seqC_gen.generate(seqC_MS_list,
                                  seqC_dur_max,
                                  seqC_sample_size,
                                  add_zero=True,
                                  single_dur=dur,
                                  )
        print(f'generated seqC shape', seqCs.shape)

        seqCs, theta, probR = simulate_dataset(
            seqCs=seqCs,
            prior=prior,
            num_prior_sample=num_prior_sample,
            modelName=model_name,
            use_seqC_summary=False,
            summary_length=8,
            # nan2num=nan2num,
            # num_LR_sample=num_LR_sample,
            num_workers=16,
        )

        print(f'seqCs.shape: {seqCs.shape}, theta.shape: {theta.shape}, probR.shape: {probR.shape}')

        # save the dataset in a hdf5 file
        seqC_group.create_dataset(f'seqC_dur{dur + 1}', data=seqCs)
        theta_group.create_dataset(f'theta_dur{dur + 1}', data=theta)
        probR_group.create_dataset(f'probR_dur{dur + 1}', data=probR)

        print(f'data dur {dur + 1} written to the file {save_data_dir}\n')

    f.close()
    print(f'data written to the file {save_data_dir}')


def prepare_training_data_from_sampled_Rchoices(
        dataset_dir: str or Path,
        dur_list: list,
        nan2num=-1,
        num_probR_sample=30,
):
    """ preparing the dataset (x, theta) pair for training
    from (seqC, theta, probR) -> (x, theta)
    x is the composition of normalized sequence C and sampled probR choice

    Args:
        dataset_dir: (string) the path to the dataset
        dur_list:   (list) the list of durations to be used
        nan2num:   (float) the value to replace nan in seqC
        num_probR_sample: (int) the number of samples to be drawn from probR

    Returns:
        x: (np.array) the input data for training
        theta: (np.array) the theta for training
    """

    f = h5py.File(dataset_dir, 'r')
    seqC_group = f['seqC_group']
    theta_group = f['theta_group']
    probR_group = f['probR_group']
    single_seqC_shape = seqC_group[f'seqC_dur{3}'].shape
    single_theta_shape = theta_group[f'theta_dur{3}'].shape
    single_probR_shape = probR_group[f'probR_dur{3}'].shape

    seqC_shape = [single_seqC_shape[0] * len(dur_list) * num_probR_sample, single_seqC_shape[1]]
    theta_shape = [single_theta_shape[0] * len(dur_list) * num_probR_sample, single_theta_shape[1]]
    RChoice_shape = [single_theta_shape[0] * len(dur_list) * num_probR_sample, 1]

    seqC, theta, RChoice = np.empty(seqC_shape), np.empty(theta_shape), np.empty(RChoice_shape)

    print('loading data...')
    start_time = time.time()
    for i, dur in tqdm(enumerate(dur_list)):
        seqC_part = np.repeat(seqC_group[f'seqC_dur{dur}'][:], num_probR_sample, axis=0)
        theta_part = np.repeat(theta_group[f'theta_dur{dur}'][:], num_probR_sample, axis=0)
        probR = probR_group[f'probR_dur{dur}'][:]
        RChoice_part = np.array([
            np.random.choice([0, 1], size=num_probR_sample, p=[1 - prob[0], prob[0]])
            for prob in probR
        ]).reshape(-1, 1)

        seqC[i * single_seqC_shape[0] * num_probR_sample
             : (i + 1) * single_seqC_shape[0] * num_probR_sample, :] = seqC_part
        theta[i * single_theta_shape[0] * num_probR_sample
              : (i + 1) * single_theta_shape[0] * num_probR_sample, :] = theta_part
        RChoice[i * single_theta_shape[0] * num_probR_sample
                : (i + 1) * single_theta_shape[0] * num_probR_sample, :] = RChoice_part
    print(f'data loaded, time used: {time.time() - start_time:.2f}s')

    # adjust seqC -> nan2num + normalize
    seqC = np.nan_to_num(seqC, nan=nan2num)
    seqC = (seqC - nan2num) / (1 - nan2num)
    # seqC[0,:]
    x = np.concatenate([seqC, RChoice], axis=1)
    # x.shape, seqC.shape, theta.shape, RChoice.shape
    f.close()

    return torch.tensor(x, dtype=torch.float32), torch.tensor(theta, dtype=torch.float32)


if __name__ == '__main__':
    # remember to generate the cython code first
    # generate the sequence C, theta, probR dataset

    simulate_and_store(
        save_data_dir='../data/training_datasets/training_dataset_test.hdf5',

        seqC_MS_list=[0.2, 0.4, 0.8],
        seqC_dur_max=14,  # 2, 4, 6, 8, 10, 12, 14 -> 7
        seqC_sample_size=700,

        prior_min=[-3.7, -36, 0, -34, 5],
        prior_max=[2.5, 71, 0, 18, 7],
        num_prior_sample=500,

        model_name='B-G-L0S-O-N-',

        test=True,
    )

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
    )