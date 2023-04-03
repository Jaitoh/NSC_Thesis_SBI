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
from utils.set_seed import setup_seed

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

def one_DM_simulation_and_output_figure(seqC, params, model_name, figure_name):

    # check seqC and params dimension should be 1
    if len(seqC.shape) != 1 or len(params.shape) != 1:
        raise ValueError('seqC and params dimension should be 1')

    params_ = copy.deepcopy(params)
    seqC_ = copy.deepcopy(seqC)

    model = DM_model(params=params_, model_name=model_name)
    a, probR = model.simulate(np.array(seqC_))

    fig = plot_a(a, probR, figure_name)

    return (seqC, params, probR, fig)


def one_DM_simulation(seqC, params, model_name, i, j, k, l):
    """ do one simulation of DM model with one seqC and one param input, returns probR
    """

    # check seqC and params dimension should be 1
    if len(seqC.shape) != 1 or len(params.shape) != 1:
        raise ValueError('seqC and params dimension should be 1')

    model = DM_model(params=params, model_name=model_name)
    _, probR = model.simulate(np.array(seqC))

    return seqC, params, probR, i, j, k, l


def DM_sim_for_seqCs_parallel(
        seqCs,
        prior,
        num_prior_sample,
        model_name='B-G-L0S-O-N-',
        num_workers=16,
        save_data_path=None,
):
    """sample params from prior and simulate probR with DM model with multiple seqCs inputs
    
    Args:   
        seqCs:              input sequences of shape [dur_len, MS_len, sample_size, 15] e.g. [7, 3, 700, 15]
        prior:              prior distribution
        num_prior_sample:   number of prior samples
        model_name:         model name
        num_workers:        number of workers for parallel processing (default: -1)
    
    Return:
        seqC:               input sequences of shape [dur_len, MS_len, sample_size, num_prior_sample, 15] e.g. [7, 3, 700, 500, 15]
        theta:              parameters of shape [dur_len, MS_len, sample_size, num_prior_sample, num_params(4)]
        probR:              probability of reward of shape [dur_len, MS_len, sample_size, num_prior_sample, 1]
    
    """
    
    print(f'---\nsimulating pR with prior sample size: {num_prior_sample}, model_name: {model_name}')
    params = prior.sample((num_prior_sample,)).numpy()

    seqC  = np.empty((*seqCs.shape[:-1], params.shape[0], seqCs.shape[-1])) # [dur_len, MS_len, sample_size, num_prior_sample, 15]
    theta = np.empty((*seqCs.shape[:-1], *params.shape)) # [dur_len, MS_len, sample_size, num_prior_sample, num_params(4)]
    probR = np.empty((*seqCs.shape[:-1], params.shape[0], 1)) # [dur_len, MS_len, sample_size, num_prior_sample, 1]
    print(f'total number of simulations', np.product(probR.shape), f'with {num_workers} workers\n---')

    # limit the number of workers to the number of available cores
    tic = time.time()
    available_workers = os.cpu_count()
    if num_workers > available_workers:
        num_workers = available_workers

    # run the simulations in parallel
    # for i in range(seqCs.shape[0]):
    #     for j in range(seqCs.shape[1]):
    #         for k in range(seqCs.shape[2]):
    #             for l in range(params.shape[0]):
    #                 seqC = seqCs[i, j, k, :]
    #                 param = params[l, :]
    #                 seqC_, param_, probR_ = one_DM_simulation(seqC, param, model_name)
    #                 theta[i, j, k, l, :] = param_
    #                 probR[i, j, k, l, 0] = probR_
    
    results = Parallel(n_jobs=num_workers, verbose=1)(delayed(one_DM_simulation)(seqCs[i, j, k, :], params[l,:], model_name, i, j, k, l) \
        for i in range(seqCs.shape[0]) \
        for j in range(seqCs.shape[1]) \
        for k in range(seqCs.shape[2]) \
        for l in range(params.shape[0]))
        
    # results = Parallel(n_jobs=num_workers, verbose=1)(
    #     delayed(one_DM_simulation)(seqC, param, model_name) \
    #     for seqC, param in itertools.product(seqCs, params))
    toc = time.time()
    print(f'time elapsed for simulation: {toc - tic:.2f} seconds')

    # store the results
    print('stacking the results')
    for seqC_, param_, probR_, i, j, k, l in results:
        seqC [i, j, k, l, :] = seqC_
        theta[i, j, k, l, :] = param_
        probR[i, j, k, l, 0] = probR_
    print('done stacking the results')

    print(f'---\nComputed seqCs.shape: {seqCs.shape}, theta.shape: {theta.shape}, probR.shape: {probR.shape}')
    
    if save_data_path!=None:
        f = h5py.File(save_data_path, 'w')
        data_group = f.create_group('/data_group')
        data_group.create_dataset("seqCs", data=seqCs)
        data_group.create_dataset("theta", data=theta)
        data_group.create_dataset("probR", data=probR)
        f.close()
        print(f'Computed results written to the file {save_data_path}\n')
    
    return seqC, theta, probR


def seqC_gen_and_DM_simulate(
        
        seqC_MS_list=None,
        seqC_dur_list=None,  # 3, 5, 7, 9, 11, 13, 15 
        seqC_sample_per_MS=700,

        prior_min=None,
        prior_max=None,
        num_prior_sample=500,

        model_name='B-G-L0S-O-N-',
        
        save_data=True,
        save_data_dir='../data/training_datasets/',
        save_data_name='sim_data_seqC0_prior0_model0_',
):
    """
    simulate probR for seqCs and params, and store the dataset to the save_data_dir
    store -> seqC, params, probR

    Args:
        save_data_dir:      directory to save the dataset
        
        seqC_dur_list:      seqC duration list
        seqC_MS_list:       list of seqC_MS to generate
        seqC_sample_per_MS: number of seqC samples
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
    seqC_dur_list = [3,5,7,9,11,13,15] if seqC_dur_list is None else seqC_dur_list
    prior_min = [-3.7, 0, 0, 0, 5] if prior_min is None else prior_min
    prior_max = [2.5, 71, 0, 18, 7] if prior_max is None else prior_max

    seqC_dur_max = 15
    
    # generate prior distribution
    prior = utils.torchutils.BoxUniform(
        low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
    )
    print(f'prior sample size', num_prior_sample)
    
    # # print('---\nsetting random seed to 0')
    # setup_seed(run)
        
    print(f'---\ngenerating raw DM-{model_name} model simulation results =probR= for further training the model')
    
    if save_data:
        # test if the output folder exists
        save_data_path = Path(save_data_dir) / (save_data_name)
        with h5py.File(save_data_path, 'w') as f:
            f.create_dataset('test', data='test')
        print(f'folder/file {save_data_path} exists, it can be used to store the dataset')

        f = h5py.File(save_data_path, 'w')
        data_group = f.create_group('/data_group')
        
        info_group = f.create_group('/info_group')
        info_group.create_dataset("seqC_MS_list", data=seqC_MS_list)
        # info_group.create_dataset("seqC_dur_max", data=seqC_dur_max)
        info_group.create_dataset("seqC_sample_per_MS", data=seqC_sample_per_MS)
        info_group.create_dataset("seqC_dur_list", data=seqC_dur_list)
        info_group.create_dataset("num_prior_sample", data=num_prior_sample)
        info_group.create_dataset("prior_min", data=prior_min)
        info_group.create_dataset("prior_max", data=prior_max)
        info_group.create_dataset("model_name", data=model_name)

    print(f'---\ngenerating seqC with seqC_dur_list: {seqC_dur_list}, seqC_MS_list: {seqC_MS_list}, seqC_sample_per_MS: {seqC_sample_per_MS}')
    seqC_gen = seqC_generator()
    seqCs = seqC_gen.generate(  dur_list=seqC_dur_list,
                                MS_list=seqC_MS_list,
                                seqC_sample_per_MS=seqC_sample_per_MS,
                            )
    # print(f'generated seqC shape', seqCs.shape)
    
    print(f'---\nsimulating pR with prior sample size: {num_prior_sample}, model_name: {model_name}')
    seqCs, theta, probR = DM_sim_for_seqCs_parallel(
            seqCs=seqCs,
            prior=prior,
            num_prior_sample=num_prior_sample,
            model_name=model_name,
            num_workers=16,
        )
    print(f'---\nComputed seqCs.shape: {seqCs.shape}, theta.shape: {theta.shape}, probR.shape: {probR.shape}')
    
    if save_data:
        data_group.create_dataset("seqCs", data=seqCs)
        data_group.create_dataset("theta", data=theta)
        data_group.create_dataset("probR", data=probR)
        f.close()
        print(f'Computed results written to the file {save_data_path}\n')
    
    return seqCs, theta, probR

if __name__ == '__main__':
    # remember to generate the cython code first
    # generate the sequence C, theta, and simulate probR

    test = True

    if test:
        config = load_config(
            config_simulator_path=Path('./src/config') / 'test'/'test_simulator.yaml',
            config_dataset_path=Path('./src/config') / 'test'/'test_dataset.yaml',
            config_train_path=Path('./src/config') / 'test'/'test_train.yaml',
        )
    else:
        config = load_config(
            config_simulator_path=Path('./src/config') /'simulator'/ 'simulator_Ca_Pa_Ma.yaml',
        )

    
    seqCs, theta, probR = seqC_gen_and_DM_simulate(
        seqC_dur_list=config['x_o']['chosen_dur_list'],  # 2, 4, 6, 8, 10, 12, 14 -> 7
        seqC_MS_list=config['x_o']['chosen_MS_list'],
        seqC_sample_per_MS=config['x_o']['seqC_sample_per_MS'],

        prior_min=config['prior']['prior_min'],
        prior_max=config['prior']['prior_max'],
        num_prior_sample=config['prior']['num_prior_sample'],

        model_name=config['simulator']['model_name'],
        
        save_data=True,
        save_data_dir=config['data_dir'],
        save_data_name=config['simulator']['save_name']+'run0.h5',
    )