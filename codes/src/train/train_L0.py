"""
- dataset: offline generation 
- training: one round training
"""
import itertools
import pickle
import yaml
# import dill
import gc
# import h5py
# import yaml
# import glob
import argparse
import torch
# torch.autograd.detect_anomaly(True)
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
# from dataset.dataset import training_dataset
# from dataset.simulate_for_sbi import simulate_for_sbi
# from simulator.seqC_generator import seqC_generator
from train.MyPosteriorEstimator import MySNPE_C
from neural_nets.embedding_nets import LSTM_Embedding, LSTM_Embedding_Small, RNN_Embedding_Small
from simulator.model_sim_pR import get_boxUni_prior
from utils.get_xo import get_xo
from utils.set_seed import setup_seed, seed_worker
from utils.train import (
    print_cuda_info, 
    # choose_cat_validation_set, 
    # plot_posterior_with_label, 
    # plot_posterior_unseen,
    # train_inference_helper,
)
from utils.setup import(
    check_path, get_args, # get_args_run_from_code
)
# from utils.resource import monitor_resources
from train.MyData import collate_fn_vec

# Set the start method to 'spawn' before creating the ProcessPoolExecutor instance
# mp.set_start_method('spawn', force=True)

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
        self.data_path = Path(self.args.data_path)
        # check_path(self.log_dir, self.data_path, args)
        
        # get dataset size
        # d = len(self.config['experiment_settings']['chosen_dur_list'])
        dur_lens = [len(dur) for dur in self.config['dataset']['chosen_dur_trained_in_sequence']]
        d = max(dur_lens)
        m = len(self.config['experiment_settings']['chosen_MS_list'])
        s = self.config['experiment_settings']['seqC_sample_per_MS']
        self.dms = d*m*s
        if self.config['dataset']['seqC_process'] == 'norm':
            self.l_x = 15+1
        if self.config['dataset']['seqC_process'] == 'summary':
            if self.config['dataset']['summary_type'] == 0:
                self.l_x = 11+1
            if self.config['dataset']['summary_type'] == 1:
                self.l_x = 8+1
        
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
        if self.prior_min is None or self.prior_max is None:
            return []
        return [[x, y] for x, y in zip(self.prior_min, self.prior_max)]
    
    
    def get_neural_posterior(self):

        dms, l_x = self.dms, self.l_x

        config_density = self.config['train']['density_estimator']
        net_type = config_density['embedding_net']['type']
        
        if net_type == 'lstm':
            embedding_net = LSTM_Embedding(
                dms         = dms,
                l           = l_x,
                hidden_size = config_density['embedding_net']['hidden_size'],
                output_size = config_density['embedding_net']['output_size'],
            )
        
        if net_type == 'lstm_small':
            embedding_net = LSTM_Embedding_Small(
                dms         = dms,
                l           = l_x,
                hidden_size = config_density['embedding_net']['hidden_size'],
                output_size = config_density['embedding_net']['output_size'],
            )
        
        if net_type == 'rnn_small':
            embedding_net = RNN_Embedding_Small(
                dms         = dms,
                l           = l_x,
                hidden_size = config_density['embedding_net']['hidden_size'],
                output_size = config_density['embedding_net']['output_size'],
            )

        neural_posterior = posterior_nn(
            model           = config_density['posterior_nn']['model'],
            embedding_net   = embedding_net, # type: ignore
            hidden_features = config_density['posterior_nn']['hidden_features'],
            num_transforms  = config_density['posterior_nn']['num_transforms'],
        )

        return neural_posterior


    # def posterior_analysis(self, posterior, current_round, run):
    #     # posterior analysis
    #     print(f"\n--- posterior sampling ---")
    #     for fig_idx in tqdm(range(len(self.post_val_set['x']))):
            
    #         fig_x, _ = plot_posterior_with_label(
    #             posterior       = posterior, 
    #             sample_num      = self.config['train']['posterior']['sampling_num'],
    #             x               = self.post_val_set['x'][fig_idx].to(self.device),
    #             true_params     = self.post_val_set['theta'][fig_idx],
    #             limits          = self._get_limits(),
    #             prior_labels    = self.config['prior']['prior_labels'],
    #         )
    #         plt.savefig(f"{self.log_dir}/posterior/figures/post_plot_x_val_{fig_idx}_round{current_round}_run{run}.png")
    #         plt.close(fig_x)
    #         fig_x_shuffle, _ = plot_posterior_with_label(
    #             posterior       = posterior, 
    #             sample_num      = self.config['train']['posterior']['sampling_num'],
    #             x               = self.post_val_set['x_shuffled'][fig_idx].to(self.device),
    #             true_params     = self.post_val_set['theta'][fig_idx],
    #             limits          = self._get_limits(),
    #             prior_labels    = self.config['prior']['prior_labels'],
    #         )
    #         plt.savefig(f"{self.log_dir}/posterior/figures/post_plot_x_val_shuffled_{fig_idx}_round{current_round}_run{run}.png")
        
    #     # save posterior for each round and run using pickle
    #     with open(f"{self.log_dir}/posterior/figures/posterior_round{current_round}_run{run}.pkl", 'wb') as f:
    #         pickle.dump(posterior, f)
            
    #     # check posterior for x_o
    #     fig, _ = plot_posterior_unseen(
    #         posterior       = posterior, 
    #         sample_num      = self.config['train']['posterior']['sampling_num'],
    #         x               = self.x_o.to(self.device),
    #         limits          = self._get_limits(),
    #         prior_labels    = self.config['prior']['prior_labels'],
    #     )
    #     plt.savefig(f"{self.log_dir}/posterior/figures/post_plot_x_o_round{current_round}_run{run}.png")

    def get_my_data_kwargs(self):
        
        my_dataset_kwargs = {
            'data_path'                      : self.args.data_path,
            'config'                         : self.config,
            'chosen_dur_trained_in_sequence' : self.config['dataset']['chosen_dur_trained_in_sequence'],
            'validation_fraction'            : self.config['dataset']['validation_fraction'],
            'use_data_prefetcher'            : self.config['dataset']['use_data_prefetcher'],
            'num_train_sets'                 : self.config['dataset']['num_train_sets'],
            'crop_dur'                       : self.config['dataset']['crop_dur'],
            'num_max_sets'                   : self.config['dataset']['num_max_sets'],
        }
        
        if self.config['dataset']['batch_process_method'] == 'collate_fn':
            
            my_dataloader_kwargs = {
                'num_workers'       :  self.config['dataset']['num_workers'],
                'worker_init_fn'    :  seed_worker,
                'collate_fn'        :  lambda batch: collate_fn_vec(batch=batch, config=self.config, shuffling_method=self.config['dataset']['shuffling_method']),
                'prefetch_factor'   :  self.config['dataset']['prefetch_factor'],
            } 
        
        else: # batch_process_method == 'in_dataset'
            
            my_dataloader_kwargs = {
                'num_workers'   :  self.config['dataset']['num_workers'],
                # 'batch_size'    :  self.config['dataset']['batch_size'],
                'worker_init_fn':  seed_worker,
            } 
            
        return my_dataloader_kwargs, my_dataset_kwargs
    
    
    def sbi_train(self, debug=False):
        # sourcery skip: boolean-if-exp-identity, remove-unnecessary-cast
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
            chosen_dur_list     = self.config['experiment_settings']['chosen_dur_list'],
            chosen_MS_list      = self.config['experiment_settings']['chosen_MS_list'],
            seqC_sample_per_MS  = self.config['experiment_settings']['seqC_sample_per_MS'],
            trial_data_path     = self.config['x_o']['trial_data_path'],
        
            seqC_process_method = self.config['dataset']['seqC_process'],
            nan2num             = self.config['dataset']['nan2num'],
            summary_type        = self.config['dataset']['summary_type'],
        )
        self.x_o = torch.tensor(x_o, dtype=torch.float32)

        # prior
        self.prior_min = self.config['prior']['prior_min']
        self.prior_max = self.config['prior']['prior_max']

        prior = utils.torchutils.BoxUniform( # type: ignore
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
        my_dataloader_kwargs, my_dataset_kwargs = self.get_my_data_kwargs()
            
        self.inference.append_simulations(
                theta         = torch.empty(1),
                x             = torch.empty(1),
                proposal      = prior,
                data_device   = 'cpu',
            )
        
        # start training
        print(f"\n======\nstart of training\n======")

        my_training_kwargs = {
            'learning_rate'        : eval(training_config['learning_rate']),
            
            'improvement_threshold': training_config['improvement_threshold'],
            'stop_after_epochs'    : training_config['stop_after_epochs'],
            'stop_after_dsets'     : training_config['stop_after_dsets'],
            
            'min_num_epochs'       : training_config['min_num_epochs'],
            'max_num_epochs'       : training_config['max_num_epochs'],
            'min_num_dsets'        : training_config['min_num_dsets'],
            'max_num_dsets'        : training_config['max_num_dsets'],
            
            'print_freq'           : training_config['print_freq'],
            'chosen_dur_trained_in_sequence': self.config['dataset']['chosen_dur_trained_in_sequence'],
            'clip_max_norm'        : eval(training_config['clip_max_norm']) if isinstance(training_config['clip_max_norm'], str) else training_config['clip_max_norm'],
            
            'num_atoms'            : training_config['num_atoms'],
            'use_combined_loss'    : True,
            'scheduler'            : training_config['scheduler'],
            'scheduler_params'     : training_config['scheduler_params'],
            
        }
        
        # run training with current run updated dataset
        self.inference, density_estimator = self.inference.train( # type: ignore
            log_dir                 = self.log_dir,
            config                  = self.config,
            
            seed                    = self.args.seed,
            prior_limits            = self._get_limits(),
            
            dataset_kwargs          = my_dataset_kwargs,
            dataloader_kwargs       = my_dataloader_kwargs,
            training_kwargs         = my_training_kwargs,
            
            continue_from_checkpoint= self.args.continue_from_checkpoint,
            debug                   = debug,
        )  # density estimator

            
        torch.save(deepcopy(density_estimator.state_dict()), f"{self.log_dir}/model/a_final_best_model_state_dict.pt")
        
        print(f"---\nfinished training in {(time.time()-start_time_total)/60:.2f} min")

    
def main():  # sourcery skip: extract-method
    args = get_args()
    print(args.log_dir)
    
    log_dir = Path(args.log_dir)
    data_path = Path(args.data_path)
    check_path(log_dir, data_path, args)
    
    # monitor resources usage
    PID = os.getpid()
    print(f"PID: {PID}")
    # log_file = f"{args.log_dir}/resource_usage.log"
    # monitor_process = multiprocessing.Process(target=monitor_resources, args=(PID, 5, log_file))
    # monitor_process.start()
    
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
        solver.sbi_train(debug=args.debug)
    
    # except Exception as e:
    #         print(f"An error occurred: {e}")
    finally:
        
        del solver
        gc.collect()
        print('solver deleted')
        torch.cuda.empty_cache()
        print('cuda cache emptied')
        # del solver
        # print('solver deleted')
        


if __name__ == '__main__':
    main()

