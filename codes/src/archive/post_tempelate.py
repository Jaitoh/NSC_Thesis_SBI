"""
generate/prepare training dataset for sbi inference: x, theta
"""
import time
import sys
sys.path.append('./src')

import numpy as np
import torch

from simulator.DM_model import DM_model
from simulator.seqC_pattern_summary import seqC_pattern_summary
from data_generator.input_c import seqCGenerator

from sbi import utils as utils
from joblib import Parallel, delayed
import itertools

import h5py
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
    return np.vstack([seqC]*n_repeat)

def simulate(args):

    seqC, params, modelName, num_LR_sample, nan2num = args
    # print(seqC, params)
    
    model = DM_model(params=params, modelName=modelName)
    seqC     = seqC.reshape(1, -1)
    _, probR = model.simulate(np.array(seqC))
    # print(f'probR: {probR}')
    cDist = np.random.choice([0,1], p=[1-probR, probR], size=num_LR_sample) # 0: left, 1: right

    seqC_hat = seq_norm_repeat(seqC, nan2num=nan2num, n_repeat=num_LR_sample)
    
    x_ = torch.as_tensor(np.hstack([seqC_hat, cDist.reshape(-1, 1)]))

    return (x_, torch.as_tensor(np.vstack([params]*num_LR_sample)))


def simulate_with_summary(args):
    
    seqC, params, modelName, num_LR_sample, nan2num  = args
    # print(seqC, params)
    
    model    = DM_model(params, modelName=modelName)
    seqC     = seqC.reshape(1, -1)
    _, probR = model.stoch_simulation(seqC)
    
    seqC_hat = seqC_pattern_summary(seqC).reshape(-1)
    x_       = torch.as_tensor(np.append(seqC_hat, probR))
    
    return (x_, torch.as_tensor(params))

def prepare_dataset_for_training(
        seqCs,
        prior,
        num_prior_sample,
        modelName='B-G-L0S-O-N-',
        use_seqC_summary=False,
        summary_length=5,
        nan2num=-2,
        num_LR_sample=100,
        num_workers=-1
        ):
    """generate dataset for training sbi inference
    
    Args:   
        seqCs:          input sequences
        prior:          prior distribution
        prior_sample:   number of sample from prior
        use_seqC_summary (bool, optional): whether to use summary of seqC. Defaults to False.
        
    """
    
    params = prior.sample((num_prior_sample,)).numpy()
    
    simulate_func   = simulate_with_summary if use_seqC_summary else simulate
    dataset_size    = len(seqCs) * num_prior_sample if use_seqC_summary else len(seqCs) * num_prior_sample * num_LR_sample
    x       = torch.empty((dataset_size, summary_length+1 if use_seqC_summary else seqCs.shape[1]+1)) 
    theta   = torch.empty((dataset_size, params.shape[1]))
    # x       = torch.empty((0, summary_length+1 if use_seqC_summary else seqCs.shape[1]+1)) # 1 stands for the probR
    # theta   = torch.empty((0, params.shape[1]))
    
    print(f'number of simuations', len(seqCs) * num_prior_sample)
    # for i, (seqC, param) in enumerate(itertools.product(seqCs, params)):
    #     # print(seqC, param)
    #     x_, theta_ = simulate_func((seqC, param, modelName, num_LR_sample, nan2num))
    #     x[i*x_.shape[0]:(i+1)*x_.shape[0], :]     = x_
    #     theta[i*theta_.shape[0]:(i+1)*theta_.shape[0], :]     = theta_
    #     progress_bar(i, dataset_size)
    tic = time.time()
    results = Parallel(n_jobs=num_workers, verbose=1)(
        delayed(simulate_func)((seqC, param, modelName, num_LR_sample, nan2num)) \
        for seqC, param in itertools.product(seqCs, params))
    toc = time.time()
    print(f'time elapsed for simulation: {toc-tic:.2f} seconds')
    
    print('stacking the results')
    len_results = len(results)
    tic = time.time()
    
    for i, (x_, theta_) in enumerate(results):
        x[i*x_.shape[0]:(i+1)*x_.shape[0], :]     = x_
        theta[i*theta_.shape[0]:(i+1)*theta_.shape[0], :]     = theta_
        progress_bar(i, len_results)
    
    toc = time.time()
    print(f'time elapsed for storing: {toc-tic:.2f} seconds')
    
    # print(f'x: {x}, theta: {theta}')
    return x, theta

def progress_bar(i, total):
    # print when the progress is 2, 4, 6, ... 100%
    if i % int((total * 0.5)) == 0:
        print(f'{i/total*100:.2f}%', end='\r')
        
    
if __name__ == '__main__':
    
    # test if the output folder exists
    save_data_dir = '../data/training_dataset_parts.hdf5'
    
    # generate seqC input sequence
    seqC_MS_list = [0.2, 0.4, 0.8]
    seqC_dur_max = 14 # 2, 4, 6, 8, 10, 12, 14 -> 7
    seqC_sample_size = 700*100
    seqC_sample_size = 700
    # seqC_sample_size = 10 # test
    
    seqC_gen = seqCGenerator()
    seqCs = seqC_gen.generate(seqC_MS_list,
                            seqC_dur_max,
                            seqC_sample_size,
                            add_zero=True
                            )
    print(f'generated seqC shape', seqCs.shape)
    
    # generate prior distribution
    prior_min = [-3.7, -36, 0, -34, 5]
    prior_max = [2.5,  71,  0,  18, 7]
    prior = utils.torchutils.BoxUniform(
        low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
    )
    num_prior_sample = int(10**(len(prior_min)-1)) # 10000 in this case
    num_prior_sample = 500
    # num_prior_sample = 100
    # num_prior_sample = 10 # test 
    print(f'prior sample size', num_prior_sample)
    
    
    # generate dataset
    model_name      = 'B-G-L0S-O-N-'
    nan2num         = -2
    num_LR_sample   = 10
    # num_LR_sample   = 5
    # num_LR_sample   = 2 # test
    print(f'number of probR sampling: {num_LR_sample}')
    
    n_split = 50
    seqCs = np.array_split(seqCs, n_split)
    seqC_len = np.repeat(np.arange(3, seqC_dur_max+2, 2), seqC_sample_size)
    seqC_lens = np.array_split(seqC_len, n_split)
    
    f = h5py.File(save_data_dir, 'a')
    seqC_group  = f.create_group('/seqC_group')
    f.create_dataset('info', data={
        "seqC_MS_list": seqC_MS_list,
        "seqC_dur_max": seqC_dur_max,
        "seqC_sample_size": seqC_sample_size,
        "num_prior_sample": num_prior_sample,
        "prior_min": prior_min,
        "prior_max": prior_max,
        "model_name": model_name,
        "nan2num": nan2num,
        "num_LR_sample": num_LR_sample,
        "n_split": n_split,
    })
    
    for i in range(n_split):
        
        print(f'\nprocessing {i+1}/{n_split}...')
        
        seqC = seqCs[i]
        print(f'processed seqC shape', seqC.shape)
        
        seqC_group.create_dataset(f'seqC_part{i}', data=seqC)
        seqC_group.create_dataset(f'seqC_len_part{i}', data=seqC_lens[i])
        
        print(f'data part {i+1}/{n_split} written to the file {save_data_dir}')
    
    f.close()
    print(f'data written to the file {save_data_dir}')