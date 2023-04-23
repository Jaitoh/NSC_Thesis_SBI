"""
- dataset: online generation 
- training: multi-round
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
from codes.src.train.MyDataset import collate_fn_probR
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
        seed = args.seed
        setup_seed(seed)
        # self.g = torch.Generator()
        # self.g.manual_seed(seed)

        self.prior              = None
        self.posterior          = None
        self.density_estimator  = None
        self.prior_max          = None
        self.prior_min          = None
        self.inference          = None



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


    def posterior_analysis(self, posterior, current_round, run):
        # posterior analysis
        print(f"\n--- posterior sampling ---")
        for fig_idx in tqdm(range(len(self.post_val_set['x']))):
            
            fig_x, _ = plot_posterior_seen(
                posterior       = posterior, 
                sample_num      = self.config['train']['posterior']['sampling_num'],
                x               = self.post_val_set['x'][fig_idx].to(self.device),
                true_params     = self.post_val_set['theta'][fig_idx],
                limits          = self._get_limits(),
                prior_labels    = self.config['prior']['prior_labels'],
            )
            plt.savefig(f"{self.log_dir}/posterior/post_plot_x_val_{fig_idx}_round{current_round}_run{run}.png")
            plt.close(fig_x)
            fig_x_shuffle, _ = plot_posterior_seen(
                posterior       = posterior, 
                sample_num      = self.config['train']['posterior']['sampling_num'],
                x               = self.post_val_set['x_shuffled'][fig_idx].to(self.device),
                true_params     = self.post_val_set['theta'][fig_idx],
                limits          = self._get_limits(),
                prior_labels    = self.config['prior']['prior_labels'],
            )
            plt.savefig(f"{self.log_dir}/posterior/post_plot_x_val_shuffled_{fig_idx}_round{current_round}_run{run}.png")
        
        # save posterior for each round and run using pickle
        with open(f"{self.log_dir}/posterior/posterior_round{current_round}_run{run}.pkl", 'wb') as f:
            pickle.dump(posterior, f)
            
        # check posterior for x_o
        fig, _ = plot_posterior_unseen(
            posterior       = posterior, 
            sample_num      = self.config['train']['posterior']['sampling_num'],
            x               = self.x_o.to(self.device),
            limits          = self._get_limits(),
            prior_labels    = self.config['prior']['prior_labels'],
        )
        plt.savefig(f"{self.log_dir}/posterior/post_plot_x_o_round{current_round}_run{run}.png")

    def get_my_dataloader_kwargs(self, Rchoice_method):
        
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
            } # TODO 简化成 1个 arg 同时将 probR_sampling method 建立 collate_fn 以减少MEM 使用
        else:
            my_dataloader_kwargs = {
                # 'num_workers': training_config['num_workers'],
                'worker_init_fn':  seed_worker,
                # 'generator':   self.g,
            }
            
            
        return my_dataloader_kwargs
    
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
        my_dataloader_kwargs = self.get_my_dataloader_kwargs(Rchoice_method)
        
        if self.gpu:
            my_dataloader_kwargs['pin_memory'] = True

        self.post_val_set = {
            "x"             : torch.empty((0, self.dms, self.l_x)),
            "x_shuffled"    : torch.empty((0, self.dms, self.l_x)),
            "theta"         : torch.empty((0, self.l_theta)),
        }
        
        # start training
        for current_round in [0,]:
            
            # get simulated data
            x, theta = simulate_for_sbi(
                proposal        = proposal,
                config          = self.config,
            )
            
            # choose and update the validation set
            if len(self.post_val_set['x']) <= 5:
                self.post_val_set = choose_cat_validation_set(
                    x               = x, 
                    theta           = theta, 
                    val_set_size    = self.config['train']['posterior']['val_set_size'],
                    post_val_set    = self.post_val_set,
                )
            
            # append simulated data to "current round" dataset
            print('appending simulated data to current round dataset')
            self.inference.append_simulations(
                theta         = theta,
                x             = x,
                proposal      = proposal,
                data_device   = 'cpu',
            )
            
            # train for multiple runs
            for run in range(training_config['num_runs']):

                print(f"\n======\nstart of training round {current_round} run {run}/{training_config['num_runs']-1}\n======")

                # print(f"---\nstart training")
                start_time = time.time()
                
                # save x, theta for each round and run
                if self.config['dataset']['save_train_data']:
                    
                    torch.save(x, f'{self.log_dir}/training_dataset/x_round{current_round}_run{run}.pt')
                    torch.save(theta, f'{self.log_dir}/training_dataset/theta_round{current_round}_run{run}.pt')
                    
                    print(f'x and theta saved to {self.log_dir}/training_dataset')
                
                # run training with current run updated dataset
                self.inference, density_estimator = self.inference.train(
                    # num_atoms               = training_config['num_atoms'],
                    training_batch_size     = training_config['training_batch_size'],
                    learning_rate           = eval(training_config['learning_rate']),
                    validation_fraction     = training_config['validation_fraction'],
                    stop_after_epochs       = training_config['stop_after_epochs'],
                    # max_num_epochs          = training_config['max_num_epochs'],
                    clip_max_norm           = training_config['clip_max_norm'],
                    calibration_kernel      = None,
                    resume_training         = (run!=0) or (current_round!=0), # resume training if not the first run
                    force_first_round_loss  = True if current_round==0 else False,
                    discard_prior_samples   = False,
                    use_combined_loss       = True,
                    retrain_from_scratch    = False,
                    show_train_summary      = True,
                    seed                    = self.args.seed,
                    dataloader_kwargs       = my_dataloader_kwargs,
                )  # density estimator

                # save best model for each round and run
                best_model_state_dict = self.inference._best_model_state_dict
                torch.save(best_model_state_dict, f"{self.log_dir}/model/best_model_round{current_round}_run{run}.pt")
                
                print(f'finished training of === round {current_round} run {run} === in {(time.time()-start_time)/60:.2f} min\n\n')
                
                
                # posterior analysis
                posterior = self.inference.build_posterior(density_estimator)
                self.posterior.append(posterior)
                self.posterior_analysis(posterior, current_round, run)
                
                
                # if not the last run
                # run simulation during training 
                # append to existing dataset after training
                if run != training_config['num_runs']-1:
                    
                    x, theta = simulate_for_sbi(
                        proposal        = proposal,
                        config          = self.config,
                    )
                    
                    # choose and update the validation set
                    if len(self.post_val_set['x']) <= 5:
                        self.post_val_set = choose_cat_validation_set(
                            x               = x, 
                            theta           = theta, 
                            val_set_size    = self.config['train']['posterior']['val_set_size'],
                            post_val_set    = self.post_val_set,
                        )
                    
                    self.inference.append_simulations_for_run(
                        theta = theta,
                        x = x,
                        current_round = current_round,
                        data_device = 'cpu',
                    )
                
                else:
                    
                    print(f"---\nfinished training of {training_config['num_runs']} runs in {(time.time()-start_time_total)/60:.2f} min")
                    proposal = posterior.set_default_x(x_o)
                
                
        # save post_val_set for after all runs
        with open(f"{self.log_dir}/posterior/post_val_set.pkl", 'wb') as f:
            pickle.dump(self.post_val_set, f)

                            
        print(f"finished training of {training_config['num_rounds']} rounds each of {training_config['num_runs']} runs in {(time.time()-start_time_total)/60:.2f} min")


    def save_model(self):

        print('---\nsaving model...')

        inference_dir           = self.log_dir / 'inference.pkl'
        with open(inference_dir, 'wb') as f:
            pickle.dump(self.inference, f)
            
        print('inference saved to: ',           inference_dir)

        # density_estimator_dir   = self.log_dir / 'density_estimator.pkl'
        # with open(density_estimator_dir, 'wb') as f:
        #     pickle.dump(self.density_estimator, f)

        # posterior_dir           = self.log_dir / 'posterior.pkl'
        # with open(posterior_dir, 'wb') as f:
        #     pickle.dump(self.posterior, f)

        # print('density_estimator saved to: ',   density_estimator_dir)
        # print('posterior saved to: ',           posterior_dir)



def main():
    args = get_args()
    
    # monitor resources usage
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

        solver = Solver(args, config)
        solver.sbi_train()
        solver.save_model()
        
        # # save the solver
        # with open(Path(args.log_dir) / 'solver.pkl', 'wb') as f:
        #     # pickle.dump(solver, f)
        #     dill.dump(solver, f)
        # print(f'solver saved to: {Path(args.log_dir) / "solver.pkl"}')

    finally:
        monitor_process.terminate()



if __name__ == '__main__':
    main()

