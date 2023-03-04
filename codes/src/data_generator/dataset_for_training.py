"""
generate/prepare training dataset for sbi inference: x, theta
"""

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
    # x       = torch.empty((dataset_size, summary_length+1 if use_seqC_summary else seqCs.shape[1]+1)) 
    # theta   = torch.empty((dataset_size, params.shape[1]))
    x       = torch.empty((0, summary_length+1 if use_seqC_summary else seqCs.shape[1]+1)) # 1 stands for the probR
    theta   = torch.empty((0, params.shape[1]))
    
    print(f'number of simuations', dataset_size)
    # for i, (seqC, param) in enumerate(itertools.product(seqCs, params)):
    #     # print(seqC, param)
    #     x_, theta_ = simulate_func((seqC, param, modelName, num_LR_sample, nan2num))
    results = Parallel(n_jobs=num_workers, verbose=1)(
        delayed(simulate_func)((seqC, param, modelName, num_LR_sample, nan2num)) \
        for seqC, param in itertools.product(seqCs, params))
    
    # vertical stack the results
    for i, (x_, theta_) in enumerate(results):
        x      = torch.vstack([x, x_])
        theta  = torch.vstack([theta, theta_])        
        # x[i, :]     = x_
        # theta[i, :] = theta_
    
    # print(f'x: {x}, theta: {theta}')
    return x, theta
    
if __name__ == '__main__':
    
    # test if the output folder exists
    save_data_dir = '../data/training_dataset.hdf5'
    with h5py.File(save_data_dir, 'w') as f:
        f.create_dataset('test', data='test')
    print(f'folder {save_data_dir} exists')
    
    # generate seqC input sequence
    seqC_MS_list = [0.2, 0.4, 0.8]
    seqC_dur_max = 14 # 2, 4, 6, 8, 10, 12, 14 -> 7
    seqC_sample_size = 700*100
    seqC_sample_size = 700
    seqC_sample_size = 10 # test
    
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
    num_prior_sample = 1000
    num_prior_sample = 100
    num_prior_sample = 10 # test 
    print(f'prior sample size', num_prior_sample)
    
    
    # generate dataset
    model_name      = 'B-G-L0S-O-N-'
    nan2num         = -2
    num_LR_sample   = 10
    num_LR_sample   = 5
    # num_LR_sample   = 2 # test
    
    x, theta = prepare_dataset_for_training(
        seqCs=seqCs,
        prior=prior,
        num_prior_sample=num_prior_sample,
        modelName=model_name,
        use_seqC_summary=False,
        summary_length=8,
        nan2num=nan2num,
        num_LR_sample=num_LR_sample,
        num_workers=-1
    )
    
    print(x.shape, theta.shape)
    print(seqCs[0], x[0], theta[0])
    
    # save the dataset in a hdf5 file
    with h5py.File(save_data_dir, 'w') as f:
        f.create_dataset('x', data=x)
        f.create_dataset('theta', data=theta)
    