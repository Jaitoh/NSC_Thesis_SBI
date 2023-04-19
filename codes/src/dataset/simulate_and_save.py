"""
using cnn to parse input sequence of shape x(batch_size, D,M,S,T,C, L_x) theta(D,M,S,T,C, L_theta)
and output the probability of each base
"""
import itertools
import pickle
import yaml
# import dill

# import h5py
# import yaml
# import glob
import argparse
import torch
import time
import os
import multiprocessing
import numpy as np
from pathlib import Path
from copy import deepcopy
from typing import Any, Callable
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor

from sbi import analysis
from sbi import utils as utils
from sbi.utils.get_nn_models import posterior_nn
from torch.utils.tensorboard import SummaryWriter

import sys

sys.path.append('./src')
# from dataset.dataset_generator import simulate_and_store, prepare_training_data_from_sampled_Rchoices
# from dataset.seqC_generator import seqC_generator
from config.load_config import load_config
from dataset.dataset import training_dataset
from dataset.simulate_for_sbi import simulate_for_sbi
from simulator.seqC_generator import seqC_generator
from train.collate_fn import collate_fn_probR
from train.MySNPE_C import MySNPE_C
from neural_nets.embedding_nets import LSTM_Embedding
from simulator.model_sim_pR import get_boxUni_prior
from utils.get_xo import get_xo
from utils.set_seed import setup_seed, seed_worker
from utils.train import (
    get_args, print_cuda_info, choose_cat_validation_set, 
    plot_posterior_seen, plot_posterior_unseen,
    check_path, train_inference_helper,
)
from utils.resource import monitor_resources

# Set the start method to 'spawn' before creating the ProcessPoolExecutor instance
mp.set_start_method('spawn', force=True)

class simulator:
    
    def __init__(self, args, config):

        self.args = args
        self.config = config
        
        self.log_dir = Path(self.args.log_dir)
        self.data_dir = Path(config['data_dir'])
        check_path(self.log_dir, self.data_dir, args)
        
        # get dataset size
        d = len(self.config['x_o']['chosen_dur_list'])
        m = len(self.config['x_o']['chosen_MS_list'])
        s = self.config['x_o']['seqC_sample_per_MS']
        self.dms = d*m*s
        self.l_x = 15+1
        self.l_theta = len(self.config['prior']['prior_min'])
        
        
        # save the config file using yaml
        yaml_path = Path(self.log_dir) / 'config.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f)
        print(f'config file saved to: {yaml_path}')

        # set seed
        seed = args.seed + args.run
        setup_seed(seed)
        # self.g = torch.Generator()
        # self.g.manual_seed(seed)

    def simulate_and_save(self, run):
        """
        train the sbi model

        Args:
            x     (torch.tensor): shape (T*C, D*M*S, L_x)
            theta (torch.tensor): shape (T*C, L_theta)
        """
 
        # prior
        self.prior_min = self.config['prior']['prior_min']
        self.prior_max = self.config['prior']['prior_max']

        prior = utils.torchutils.BoxUniform(
            low     = np.array(self.prior_min, dtype=np.float32),
            high    = np.array(self.prior_max, dtype=np.float32),
        )
        
        # get simulated data
        x, theta = simulate_for_sbi(
            proposal        = prior,
            config          = self.config,
        )
        
        torch.save(x, f'{self.log_dir}/training_dataset/x_round{0}_run{run}.pt')
        torch.save(theta, f'{self.log_dir}/training_dataset/theta_round{0}_run{run}.pt')
    

def main():
    args = get_args()
    
    PID = os.getpid()
    print(f"PID: {PID}")
    log_file = f"{args.log_dir}/resource_usage.log"
    monitor_process = multiprocessing.Process(target=monitor_resources, args=(PID, 5, log_file))
    monitor_process.start()
    
    try:
            
        config = load_config(
            config_simulator_path=args.config_simulator_path,
            config_dataset_path=args.config_dataset_path,
            config_train_path=args.config_train_path,
        )

        print(f'\n--- args ---')
        for arg, value in vars(args).items():
            print(f'{arg}: {value}')

        print('\n--- config keys ---')
        print(config.keys())

        sim = simulator(args, config)
        sim.simulate_and_save(run=args.run)
        
    finally:
        
        monitor_process.terminate()



if __name__ == '__main__':
    main()

