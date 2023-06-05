"""
- dataset: offline generation 
- training: one round training
"""
import itertools
import datetime
import pickle
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
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
# from config.load_config import load_config
# from dataset.dataset import training_dataset
# from dataset.simulate_for_sbi import simulate_for_sbi
# from simulator.seqC_generator import seqC_generator
from train.MyPosteriorEstimator import MySNPE_C
from neural_nets.embedding_nets import LSTM_Embedding, LSTM_Embedding_Small, RNN_Embedding_Small, Conv1D_RNN, RNN_Multi_Head
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
from train.MyData import collate_fn_vec, collate_fn_vec_high_dim

# Set the start method to 'spawn' before creating the ProcessPoolExecutor instance
# mp.set_start_method('spawn', force=True)

class Solver:
    """
        Solver for training sbi
    """

    def __init__(self, config):

        self.config = config
        # self.test = self.config.run_test

        self.gpu = self.config.gpu and torch.cuda.is_available()
        self.device = 'cuda' if self.gpu else 'cpu'
        print(f'using device: {self.device}')
        print(f"starting time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print_cuda_info(self.device)

        self.log_dir = Path(self.config.log_dir)
        self.data_path = Path(self.config.data_path)

        # get dataset size
        # d = len(self.config['experiment_settings']['chosen_dur_list'])
        dur_lens = [len(dur) for dur in self.config.dataset.chosen_dur_trained_in_sequence]
        d = max(dur_lens)
        m = len(self.config.experiment_settings.chosen_MS_list)
        s = self.config.experiment_settings.seqC_sample_per_MS
        self.dms, self.d, self.m, self.s = d*m*s, d, m, s
        if self.config.dataset.seqC_process == 'norm':
            self.l_x = 15+1
        elif self.config.dataset.seqC_process == 'summary':
            if self.config.dataset.summary_type == 0:
                self.l_x = 11+1
            elif self.config.dataset.summary_type == 1:
                self.l_x = 8+1

        self.l_theta = len(self.config.prior.prior_min)

        # save the config file using yaml
        yaml_path = Path(self.log_dir) / 'config.yaml'
        with open(yaml_path, "w") as f:
            f.write(OmegaConf.to_yaml(config))
        print(f'config file saved to: {yaml_path}')

        # set seed
        self.seed = config.seed
        setup_seed(self.seed)
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
        
        # if net_type == 'conv1d_rnn':
        if self.config.dataset.dataset_dim == 'high_dim' and net_type == 'conv1d_rnn': #TODO check condition for using conv1d_rnn
            embedding_net = Conv1D_RNN, RNN_Multi_Head(
                DM = self.d*self.m,
                S  = self.s,
                L  = self.l_x,
            )

        if self.config.dataset.dataset_dim == 'high_dim' and net_type == 'rnn_multi_head':
            embedding_net = RNN_Multi_Head(
                DM = self.d*self.m,
                S  = self.s,
                L  = self.l_x,
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
        
        # my_dataset_kwargs = {
        #     'data_path'                      : self.config.data_path,
        #     'config'                         : self.config,
        #     'chosen_dur_trained_in_sequence' : self.config.dataset.chosen_dur_trained_in_sequence
        #     'validation_fraction'            : self.config.dataset.validation_fraction
        #     'use_data_prefetcher'            : self.config.dataset.use_data_prefetcher
        #     'num_train_sets'                 : self.config.dataset.num_train_sets
        #     'crop_dur'                       : self.config.dataset.crop_dur
        #     'num_max_sets'                   : self.config.dataset.num_max_sets
        # }

        config_dataset = self.config.dataset
        if config_dataset.batch_process_method != 'collate_fn':
            return {  # TODO check and modify in_dataset processing login, of high_dim
                'num_workers': config_dataset.num_workers,
                'worker_init_fn': seed_worker,
                'prefetch_factor': prefetch_factor if use_data_prefetcher else 2 + prefetch_factor,
                # 'batch_size'    :  self.config.dataset.batch_size
            }

        use_data_prefetcher = config_dataset.use_data_prefetcher
        prefetch_factor     = config_dataset.prefetch_factor

        collate_fn = collate_fn_vec_high_dim if config_dataset.dataset_dim == 'high_dim' else collate_fn_vec
        
        return {
            'num_workers': config_dataset.num_workers,
            'worker_init_fn': seed_worker,
            'collate_fn': lambda batch: collate_fn(
                batch=batch,
                config=self.config,
                shuffling_method=config_dataset.shuffling_method,
            ),
            'prefetch_factor': prefetch_factor if use_data_prefetcher else 2 + prefetch_factor,
        }
    
    
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
        config_x_o = self.config.x_o
        config_exp = self.config.experiment_settings
        config_dataset = self.config.dataset
        x_o = get_xo(
            subject_id          = config_x_o.subject_id,
            chosen_dur_list     = config_exp.chosen_dur_list,
            chosen_MS_list      = config_exp.chosen_MS_list,
            seqC_sample_per_MS  = config_exp.seqC_sample_per_MS,
            trial_data_path     = config_x_o.trial_data_path,
        
            seqC_process_method = config_dataset.seqC_process,
            nan2num             = config_dataset.nan2num,
            summary_type        = config_dataset.summary_type,
        )
        self.x_o = torch.tensor(x_o, dtype=torch.float32)

        # prior
        self.prior_min = self.config.prior.prior_min
        self.prior_max = self.config.prior.prior_max

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
        training_config = self.config.train.training

        # dataloader kwargs
        my_dataloader_kwargs = self.get_my_data_kwargs()
            
        self.inference.append_simulations(
                theta         = torch.empty(1),
                x             = torch.empty(1),
                proposal      = prior,
                data_device   = 'cpu',
            )
        
        # start training
        print(f"\n======\nstart of training\n======")
        
        # run training with current run updated dataset
        self.inference, density_estimator = self.inference.train( # type: ignore
            config                  = self.config,
            prior_limits            = self._get_limits(),
            dataloader_kwargs       = my_dataloader_kwargs,
            continue_from_checkpoint= self.config.continue_from_checkpoint,
            debug                   = debug,
        )  # density estimator

            
        torch.save(deepcopy(density_estimator.state_dict()), f"{self.log_dir}/model/a_final_best_model_state_dict.pt")
        
        print(f"---\nfinished training in {(time.time()-start_time_total)/60:.2f} min")

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(config : DictConfig):
    # args = get_args()
    # print(args.log_dir)
    
    # monitor resources usage
    PID = os.getpid()
    print(f"PID: {PID}")
    # log_file = f"{args.log_dir}/resource_usage.log"
    # monitor_process = multiprocessing.Process(target=monitor_resources, args=(PID, 5, log_file))
    # monitor_process.start()
    
    try:
        print('\n--- config keys ---')
        print(OmegaConf.to_yaml(config))
        # config = load_config(
        #     config_simulator_path=args.config_simulator_path,
        #     config_dataset_path=args.config_dataset_path,
        #     config_train_path=args.config_train_path,
        # )
        log_dir = Path(config.log_dir)
        data_path = Path(config.data_path)
        check_path(log_dir, data_path)
        
        solver = Solver(config)
        solver.sbi_train(debug=config.debug)

        print(f"finishing time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
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

