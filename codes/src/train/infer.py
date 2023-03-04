import sbi.inference
from sbi.utils.user_input_checks import process_prior
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import itertools
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import h5py

from joblib import Parallel, delayed
import itertools

import sys
sys.path.append('./src')
from simulator.DM_model import DM_model
from simulator.seqC_pattern_summary import seqC_pattern_summary
from parse_data.parse_trial_data import get_unique_seqC_for_subj

def simulate(args):
    seqC, params, modelName = args
    
    model    = DM_model(params, modelName=modelName)
    seqC     = seqC.reshape(1, -1)
    _, probR = model.stoch_simulation(seqC)
    
    seqCHat  = seqC_pattern_summary(seqC).reshape(-1)
    x_       = torch.as_tensor(np.append(seqCHat, probR))
    return (x_, params)
    
def prepare_data_for_sbi(prior,
                        num_sample=300,
                        subjID=1,
                        modelName='B-G-L0S-O-N-',
                        trial_data_dir='../data/trials.mat',
                        ):
    ''' prepare x, theta for sbi inference
    Args:
        prior       : prior distribution
        num_sample  : number of samples from prior
        subjID      : subject ID
        modelName   : model name
    '''
    
    _theta          = prior.sample((num_sample,)) # of shape (num_simulations, 5)
    _, uniquePulse  = get_unique_seqC_for_subj(subjID=subjID, trial_data_dir=trial_data_dir)

    num_simulations = num_sample * len(uniquePulse)
    x       = torch.zeros((num_simulations, 9))
    theta   = torch.zeros((num_simulations, 5))

    args_list = [(seqC, params, modelName) for seqC, params in itertools.product(uniquePulse, _theta)]

    with mp.Pool() as pool:
        results = list(tqdm(pool.imap(simulate, args_list), total=len(args_list)))

    for i, (x_, theta_) in enumerate(results):
        x[i, :]     = x_
        theta[i, :] = theta_
    
    return theta, x 

def prepare_data_for_sbi_useParrallel(prior,
                        num_sample=300,
                        subjID=1,
                        modelName='B-G-L0S-O-N-',
                        num_workers=8,
                        trial_data_dir='../data/trials.mat',
                        ):
    
    _theta          = prior.sample((num_sample,)) # of shape (num_simulations, 5)
    _, uniquePulse  = get_unique_seqC_for_subj(subjID=subjID, trial_data_dir=trial_data_dir)
    
    num_simulations = num_sample * len(uniquePulse)
    x       = torch.zeros((num_simulations, 9))
    theta   = torch.zeros((num_simulations, 5))
    
    results = Parallel(n_jobs=num_workers, verbose=1)(
        delayed(simulate)((seqC, params, modelName)) for seqC, params in itertools.product(uniquePulse, _theta))

    for i, (x_, params) in enumerate(results):
        x[i, :]     = x_
        theta[i, :] = params
        
    return theta, x


def my_infer(
    # simulator: Callable,
    prior,
    method: str,
    num_sample: int =300,
    # num_workers: int = 1,
    modelName: str = 'B-G-L0S-O-N-',
    subjID: int = 1,
    trial_data_dir='../data/trials.mat',
    **kwargs: Any,
):
    r"""Runs simulation-based inference and returns the posterior.

    This function provides a simple interface to run sbi. Inference is run for a single
    round and hence the returned posterior $p(\theta|x)$ can be sampled and evaluated
    for any $x$ (i.e. it is amortized).

    The scope of this function is limited to the most essential features of sbi. For
    more flexibility (e.g. multi-round inference, different density estimators) please
    use the flexible interface described here:
    https://www.mackelab.org/sbi/tutorial/02_flexible_interface/

    Args:
        prior: A probability distribution that expresses prior knowledge about the
            parameters, e.g. which ranges are meaningful for them. Any
            object with `.log_prob()`and `.sample()` (for example, a PyTorch
            distribution) can be used.
        method: What inference method to use. Either of SNPE, SNLE or SNRE.
        num_simulations: Number of simulation calls. More simulations means a longer
            runtime, but a better posterior estimate.
        num_workers: Number of parallel workers to use for simulations.
        ----------
        data_from_mem: data from memory
        save_data_dir: dir to save data to .h5 file

    Returns: Posterior over parameters conditional on observations (amortized).
    """

    try:
        method_fun: Callable = getattr(sbi.inference, method.upper())
    except AttributeError:
        raise NameError(
            "Method not available. `method` must be one of 'SNPE', 'SNLE', 'SNRE'."
        )

    # simulator, prior = prepare_for_sbi(simulator, prior)
    
    inference = method_fun(prior=prior)
    
    if 'data_from_mem' in kwargs.keys():
        theta, x = kwargs['data_from_mem']
        
    else:
        # check if folder exists by writing a test file
        if 'save_data_dir' in kwargs.keys():
            with h5py.File(kwargs['save_data_dir'], 'w') as f:
                f.create_dataset('test', data='test')
            print('folder exists')
        
        if 'num_workers' in kwargs.keys():
            theta, x = prepare_data_for_sbi_useParrallel(prior=prior,
                                    num_sample=num_sample,
                                    subjID=subjID,
                                    modelName=modelName,
                                    num_workers=kwargs['num_workers'])
        else:
            theta, x = prepare_data_for_sbi(prior=prior,
                                        num_sample=num_sample,
                                        subjID=subjID,
                                        modelName=modelName)
        
        # save data to .h5 file
        if 'save_data_dir' in kwargs.keys():
            with h5py.File(kwargs['save_data_dir'], 'w') as f:
                f.create_dataset('theta', data=theta)
                f.create_dataset('x', data=x)
            print('data saved to .h5 file')
    # theta, x = simulate_for_sbi(
    #     simulator=simulator,
    #     proposal=prior,
    #     num_simulations=num_simulations,
    #     num_workers=num_workers,
    # )
    print('start training')
    _ = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior()
    print('finished training')
    
    with h5py.File(kwargs['save_data_dir'], 'w') as f:
        f.create_dataset('posterior', data=posterior)
    print('posterior saved to .h5 file')
    
    return posterior