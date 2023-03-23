"""
generate/prepare training dataset for sbi inference: x, theta
"""
import time
import sys
import copy

sys.path.append('./src')

import os

import numpy as np
import torch

from simulator.DM_model import DM_model
from dataset.seqC_generator import seqC_generator
from config.load_config import load_config

from sbi import utils as utils
from joblib import Parallel, delayed
from pathlib import Path
import itertools

import h5py
import matplotlib.pyplot as plt

cmaps = ['tab:blue', 'tab:red', 'tab:orange', 'tab:purple']


def get_boxUni_prior(
    prior_min: np.ndarray, prior_max: np.ndarray
):
    prior = utils.torchutils.BoxUniform(
        low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
    )
    return prior


def plot_a(a, probR,
           figure_name,
           ):
    fig = plt.figure()
    # fig.suptitle('Model: ' + paramsFitted['allModelsList'][idx])
    plt.plot(a[::100], '.-', label=f'a1 probR={probR:.3f}', lw=2, color=cmaps[0])

    plt.xlabel('Time (sample)')
    plt.ylabel('a')

    lgd = plt.legend(loc = 'lower right', fontsize=24)
    # set the legend font to bold
    for text in lgd.get_texts():
        text.set_fontweight('bold')
    lgd.get_frame().set_facecolor('none')
    plt.grid(alpha=0.5)
    # change title font to bold
    plt.title(plt.title(figure_name).get_text(), fontsize=8)
    plt.close()

    return fig

def _one_DM_simulation_and_output_figure(seqC, params, model_name, figure_name):

    # check seqC and params dimension should be 1
    if len(seqC.shape) != 1 or len(params.shape) != 1:
        raise ValueError('seqC and params dimension should be 1')

    params_ = copy.deepcopy(params)
    seqC_ = copy.deepcopy(seqC)

    model = DM_model(params=params_, modelName=model_name)
    a, probR = model.simulate(np.array(seqC_))

    fig = plot_a(a, probR, figure_name)

    return (seqC, params, probR, fig)


def _one_DM_simulation(args):
    """ do one simulation of DM model with one seqC and one param input, returns probR
    """

    seqC, params, modelName = args

    # check seqC and params dimension should be 1
    if len(seqC.shape) != 1 or len(params.shape) != 1:
        raise ValueError('seqC and params dimension should be 1')

    model = DM_model(params=params, modelName=modelName)
    _, probR = model.simulate(np.array(seqC))

    return (seqC, params, probR)


def _DM_sim_for_seqCs_parallel(
        seqCs,
        prior,
        num_prior_sample,
        modelName='B-G-L0S-O-N-',
        num_workers=-1,
):
    """sample params from prior and simulate probR with DM model with multiple seqCs inputs
    
    Args:   
        seqCs:              input sequences
        prior:              prior distribution
        num_prior_sample:   number of prior samples
        modelName:          model name
        num_workers:        number of workers for parallel processing (default: -1)

    """

    params = prior.sample((num_prior_sample,)).numpy()

    simulate_func = _one_DM_simulation
    dataset_size = len(seqCs) * num_prior_sample
    seqC = np.empty((dataset_size, seqCs.shape[1]))
    theta = np.empty((dataset_size, params.shape[1]))
    probR = np.empty((dataset_size, 1))

    print(f'number of simulations', len(seqCs) * num_prior_sample)

    # limit the number of workers to the number of available cores
    tic = time.time()
    available_workers = os.cpu_count()
    if num_workers > available_workers:
        num_workers = available_workers

    # run the simulations in parallel
    results = Parallel(n_jobs=num_workers, verbose=1)(
        delayed(simulate_func)((seqC, param, modelName)) \
        for seqC, param in itertools.product(seqCs, params))
    toc = time.time()
    print(f'time elapsed for simulation: {toc - tic:.2f} seconds')

    # store the results
    print('stacking the results')
    tic = time.time()
    for i, (seqC_, theta_, probR_) in enumerate(results):
        seqC[i, :] = seqC_
        theta[i, :] = theta_
        probR[i, 0] = probR_
    toc = time.time()
    print(f'time elapsed for storing: {toc - tic:.2f} seconds')

    return seqC, theta, probR


def DM_simulate_and_store(
        save_data_dir='../data/training_datasets/training_dataset.hdf5',

        seqC_MS_list=None,
        seqC_dur_max=15,  # 3, 5, 7, 9, 11, 13, 15 -> 7
        seqC_sample_size=700,

        prior_min=None,
        prior_max=None,
        num_prior_sample=500,

        model_name='B-G-L0S-O-N-',

        # test=False,
):
    """
    simulate probR for seqCs and params, and store the dataset to the save_data_dir
    store -> seqC, params, probR

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
    prior_min = [-3.7, 0, 0, 0, 5] if prior_min is None else prior_min
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
    # num_prior_sample = 10 if test else num_prior_sample
    print(f'prior sample size', num_prior_sample)

    # generate seqC input sequence
    # seqC_sample_size = 10 if test else seqC_sample_size
    dur_list = np.arange(3, seqC_dur_max + 1, 2)

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

    for dur in dur_list:
        print(f'\nprocessing duration {dur}...')

        seqC_gen = seqC_generator()
        seqCs = seqC_gen.generate(MS_list=seqC_MS_list,
                                  dur_max=seqC_dur_max,
                                  sample_size=seqC_sample_size,
                                  single_dur=dur,
                                  )
        print(f'generated seqC shape', seqCs.shape)

        seqCs, theta, probR = _DM_sim_for_seqCs_parallel(
            seqCs=seqCs,
            prior=prior,
            num_prior_sample=num_prior_sample,
            modelName=model_name,
            num_workers=16,
        )

        print(f'seqCs.shape: {seqCs.shape}, theta.shape: {theta.shape}, probR.shape: {probR.shape}')

        # save the dataset in a hdf5 file
        seqC_group.create_dataset(f'seqC_dur{dur}', data=seqCs)
        theta_group.create_dataset(f'theta_dur{dur}', data=theta)
        probR_group.create_dataset(f'probR_dur{dur}', data=probR)

        print(f'data dur {dur} written to the file {save_data_dir}\n')

    f.close()
    print(f'data written to the file {save_data_dir}')


if __name__ == '__main__':
    # remember to generate the cython code first
    # generate the sequence C, theta, and simulate probR

    test = False

    if test:
        config = load_config(
            config_simulator_path=Path('./src/config') / 'test_simulator.yaml',
            config_dataset_path=Path('./src/config') / 'test_dataset.yaml',
            config_train_path=Path('./src/config') / 'test_train.yaml',
        )
    else:
        config = load_config(
            config_simulator_path=Path('./src/config') / 'simulator_Ca_Pa_Ma.yaml',
        )

    save_data_dir = Path(config['data_dir'])
    save_data_path = save_data_dir / config['simulator']['save_name']

    DM_simulate_and_store(
        save_data_dir=save_data_path,

        seqC_MS_list=config['seqC']['MS_list'],
        seqC_dur_max=config['seqC']['dur_max'],  # 2, 4, 6, 8, 10, 12, 14 -> 7
        seqC_sample_size=config['seqC']['sample'],

        prior_min=config['prior']['prior_min'],
        prior_max=config['prior']['prior_max'],
        num_prior_sample=config['prior']['num_prior_sample'],

        model_name=config['simulator']['model_name'],
    )