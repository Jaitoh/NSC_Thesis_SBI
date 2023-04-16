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
import os
import numpy as np
import random
from pathlib import Path
import time
import shutil
from typing import Any, Callable
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    plot_posterior_seen, plot_posterior_unseen
)

from memory_profiler import profile

@profile
class Solver:
    """
        Solver for training sbi
    """

    def __init__(self, args, config):

        self.args = args
        self.config = config
        # self.test = self.args.run_test

        self.gpu = self.args.gpu and torch.cuda.is_available()
        # self.device = torch.device('cuda') if self.gpu else torch.device('cpu')
        self.device = 'cuda' if self.gpu else 'cpu'
        print(f'using device: {self.device}')
        print_cuda_info(self.device)

        self.log_dir = Path(self.args.log_dir)
        self.data_dir = Path(config['data_dir'])
        
        
        # get dataset size
        d = len(self.config['x_o']['chosen_dur_list'])
        m = len(self.config['x_o']['chosen_MS_list'])
        s = self.config['x_o']['seqC_sample_per_MS']
        self.dms = d*m*s
        self.l_x = 15+1
        self.l_theta = len(self.config['prior']['prior_min'])
        
        
        self._check_path()
        # save the config file using yaml
        yaml_path = Path(self.log_dir) / 'config.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f)
        print(f'config file saved to: {yaml_path}')

        # set seed
        seed = args.seed
        setup_seed(seed)
        # self.g = torch.Generator()
        # self.g.manual_seed(seed)

        self.prior              = None
        self.posterior          = None
        self.density_estimator  = None
        self.prior_max    = None
        self.prior_min    = None
        self.inference          = None

    def _check_path(self):
        """
        check the path of log_dir and data_dir
        """

        print(f'\n--- dir settings ---\nlog dir: {str(self.log_dir)}')
        print(f'data dir: {str(self.data_dir)}')

        # check log path: if not exists, create; if exists, remove or a fatal error
        if not self.log_dir.exists():
            os.makedirs(str(self.log_dir))
            os.makedirs(f'{str(self.log_dir)}/model/')
            os.makedirs(f'{str(self.log_dir)}/training_dataset/')
            os.makedirs(f'{str(self.log_dir)}/posterior/')
            

        elif self.log_dir.exists() and not self.args.eval:
            if self.args.overwrite:
                shutil.rmtree(self.log_dir)
                print(f'Run dir {str(self.log_dir)} emptied.')
                os.makedirs(str(self.log_dir))
                os.makedirs(f'{str(self.log_dir)}/model/')
                os.makedirs(f'{str(self.log_dir)}/training_dataset/')
                os.makedirs(f'{str(self.log_dir)}/posterior/')
                
            else:
                assert False, f'Run dir {str(self.log_dir)} already exists.'

        # check data path, where to read the data from, exists
        if not Path(self.data_dir).exists():
            assert False, f'Data dir {str(self.data_dir)} does not exist.'


    def _get_limits(self):
        return [[x, y] for x, y in zip(self.prior_min, self.prior_max)]
    
    
    def get_neural_posterior(self):

        dms, l_x = self.dms, self.l_x

        config_density = self.config['train']['density_estimator']

        embedding_net = LSTM_Embedding(
            dms         = dms,
            l           = l_x,
            hidden_size = config_density['embedding_net']['hidden_size'],
            output_size = config_density['embedding_net']['output_size'],
        )

        neural_posterior = posterior_nn(
            model           = config_density['posterior_nn']['model'],
            embedding_net   = embedding_net,
            hidden_features = config_density['posterior_nn']['hidden_features'],
            num_transforms  = config_density['posterior_nn']['num_transforms'],
        )

        return neural_posterior


    def sbi_train(self):
        """
        train the sbi model

        Args:
            x     (torch.tensor): shape (T*C, D*M*S, L_x)
            theta (torch.tensor): shape (T*C, L_theta)
        """
        # train the sbi model
        writer = SummaryWriter(log_dir=str(self.log_dir))

        # observed data from trial experiment
        x_o = get_xo(
            subject_id          = self.config['x_o']['subject_id'],
            chosen_dur_list     = self.config['x_o']['chosen_dur_list'],
            chosen_MS_list      = self.config['x_o']['chosen_MS_list'],
            seqC_sample_per_MS  = self.config['x_o']['seqC_sample_per_MS'],
            trial_data_path     = self.config['x_o']['trial_data_path'],
        
            seqC_process_method = self.config['dataset']['seqC_process'],
            nan2num             = self.config['dataset']['nan2num'],
            summary_type        = self.config['dataset']['summary_type'],
        )
        self.x_o = torch.tensor(x_o, dtype=torch.float32)

        # prior
        self.prior_min = self.config['prior']['prior_min']
        self.prior_max = self.config['prior']['prior_max']

        prior = utils.torchutils.BoxUniform(
            low     = np.array(self.prior_min, dtype=np.float32),
            high    = np.array(self.prior_max, dtype=np.float32),
            device  = self.device,
        )
        self.prior = prior

        # get neural posterior
        neural_posterior = self.get_neural_posterior()
        print(f'neural_posterior: {neural_posterior}')
        
        self.inference = MySNPE_C(
            prior               = prior,
            density_estimator   = neural_posterior,
            device              = self.device,
            logging_level       = 'INFO',
            summary_writer      = writer,
            show_progress_bars  = True,
        )

        # print('---\ntraining property: ')
        print_cuda_info(self.device)
        start_time_total = time.time()
        self.density_estimator = []
        self.posterior = []
        proposal = prior
        training_config = self.config['train']['training']

        # dataloader kwargs
        Rchoice_method = self.config['dataset']['Rchoice_method']
        
        if Rchoice_method == 'probR':
            my_dataloader_kwargs = {
                # 'num_workers': training_config['num_workers'],
                'worker_init_fn':  seed_worker,
                # 'generator':   self.g,
                'collate_fn':  lambda batch: collate_fn_probR(
                                                batch,
                                                Rchoice_method=Rchoice_method,
                                                num_probR_sample=self.config['dataset']['num_probR_sample'],
                                            ),
            }
        else:
            my_dataloader_kwargs = {
                # 'num_workers': training_config['num_workers'],
                'worker_init_fn':  seed_worker,
                # 'generator':   self.g,
            }
            
        if self.gpu:
            my_dataloader_kwargs['pin_memory'] = True

        self.post_val_set = {
            "x"             : torch.empty((0, self.dms, self.l_x)),
            "x_shuffled"    : torch.empty((0, self.dms, self.l_x)),
            "theta"         : torch.empty((0, self.l_theta)),
        }
        
        # start training
        for current_round in range(self.config['train']['training']['num_rounds']):
            
            # get simulated data
            x, theta = simulate_for_sbi(
                proposal        = proposal,
                config          = self.config,
            )
            
            # # choose and update the validation set
            # if len(self.post_val_set['x']) <= 5:
            #     self.post_val_set = choose_cat_validation_set(
            #         x               = x, 
            #         theta           = theta, 
            #         val_set_size    = self.config['train']['posterior']['val_set_size'],
            #         post_val_set    = self.post_val_set,
            #     )
            
            # # append simulated data to "current round" dataset
            # self.inference.append_simulations(
            #     theta         = theta,
            #     x             = x,
            #     proposal      = proposal,
            #     data_device   = 'cpu',
            # )
            
            # train for multiple runs
            for run in range(training_config['num_runs']):

                print(f"\n======\nstart of round {current_round} run {run}/{training_config['num_runs']-1}\n======")

                # print(f"---\nstart training")
                start_time = time.time()
                
                # save x, theta for each round and run
                if self.config['dataset']['save_train_data']:
                    
                    torch.save(x, f'{self.log_dir}/training_dataset/x_round{current_round}_run{run}.pt')
                    torch.save(theta, f'{self.log_dir}/training_dataset/theta_round{current_round}_run{run}.pt')
                    
                    print(f'x and theta saved to {self.log_dir}/training_dataset')
                
                # if not the last run
                # run simulation during training 
                # append to existing dataset after training TODO check if the dataset size increases
                if run != training_config['num_runs']-1:
                    
                    x, theta = simulate_for_sbi(
                        proposal        = proposal,
                        config          = self.config,
                    )
                    
        
    def save_model(self):

        print('---\nsaving model...')

        inference_dir           = self.log_dir / 'inference.pkl'
        with open(inference_dir, 'wb') as f:
            pickle.dump(self.inference, f)
            
        print('inference saved to: ',           inference_dir)

        # density_estimator_dir   = self.log_dir / 'density_estimator.pkl'
        # posterior_dir           = self.log_dir / 'posterior.pkl'
        
        # with open(density_estimator_dir, 'wb') as f:
        #     pickle.dump(self.density_estimator, f)

        # with open(posterior_dir, 'wb') as f:
        #     pickle.dump(self.posterior, f)

        # print('density_estimator saved to: ',   density_estimator_dir)
        # print('posterior saved to: ',           posterior_dir)



    

def main():
    args = get_args()

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

    solver = Solver(args, config)
    solver.sbi_train()
    # solver.save_model()
    
    # # save the solver
    # with open(Path(args.log_dir) / 'solver.pkl', 'wb') as f:
    #     # pickle.dump(solver, f)
    #     dill.dump(solver, f)
    # print(f'solver saved to: {Path(args.log_dir) / "solver.pkl"}')


if __name__ == '__main__':
    main()

