import argparse
import torch
from sbi import analysis
import os
import shutil
from pathlib import Path

import os

def train_inference_helper(inference, **kwargs):
    return inference.train(**kwargs)

        
def plot_posterior_with_label(posterior, sample_num, x, true_params, limits, prior_labels):
    """ plot the posterior distribution of the seen data """
    
    samples = posterior.sample((sample_num,), x=x, show_progress_bars=False)

    fig, axes = analysis.pairplot(
        samples.cpu().numpy(),
        limits=limits,
        # ticks=[[], []],
        figsize=(10, 10),
        points=true_params.cpu().numpy(),
        points_offdiag={'markersize': 5, 'markeredgewidth': 1},
        points_colors='r',
        labels=prior_labels,
        upper=["kde"],
        diag=["kde"],
    )
    
    del samples, x, posterior
    torch.cuda.empty_cache()
    
    return fig, axes

def plot_posterior_unseen(posterior, sample_num, x, limits, prior_labels):
    """ plot the posterior distribution of the seen data """
    
    samples = posterior.sample((sample_num,), x=x, show_progress_bars=False)

    fig, axes = analysis.pairplot(
        samples.cpu().numpy(),
        limits=limits,
        # ticks=[[], []],
        figsize=(10, 10),
        labels=prior_labels,
        upper=["kde"],
        diag=["kde"],
    )
    
    del samples
    torch.cuda.empty_cache()
    
    return fig, axes
    

def choose_cat_validation_set(x, theta, val_set_size, post_val_set):
    """ choose and catenate the validation set from the input x and theta

    Args:
        x           (torch.tensor): shape (TC, DMS, L_x)
        theta       (torch.tensor): shape (TC, L_theta)
        val_set_size  (int): the size of the validation set
        post_val_set (dict): the post validation set has keys: 'x', 'x_shuffled', 'theta',

    """
    
    # randomly choose val_set_size samples from TC samples
    idx = torch.randperm(x.shape[0])
    idx_val = idx[:val_set_size]
    
    x_val       = x[idx_val,:,:]
    theta_val   = theta[idx_val,:]
    
    # randomize the order of each sequence of each line
    x_val_shuffled = torch.empty_like(x_val)
    for k in range(val_set_size): 
        x_temp = x_val[k,:,:]
        idx = torch.randperm(x_temp.shape[0])
        x_val_shuffled[k,:,:] = x_temp[idx,:] # D*M*S,L_x
    
    # append to the post validation set
    post_val_set['x']           = torch.cat((post_val_set['x'], x_val), dim=0)
    post_val_set['x_shuffled']  = torch.cat((post_val_set['x_shuffled'], x_val_shuffled), dim=0)
    post_val_set['theta']       = torch.cat((post_val_set['theta'], theta_val), dim=0)
    
    return post_val_set





def print_cuda_info(device):
    """
    Args:
        device: 'cuda' or 'cpu'
    """
    if device == 'cuda':
        print('\n--- CUDA info ---')
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
        print('--- CUDA info ---\n')
        torch.cuda.memory_summary(device=None, abbreviated=False)