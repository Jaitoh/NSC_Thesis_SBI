# TODO replacing the dataset kwargs 
import datetime
from pathlib import Path
import os
import time
import h5py
from copy import deepcopy
import numpy as np
from scipy import stats
from collections import deque
import gc
import torch
from torch import nn, ones, optim
from torch.distributions import Distribution, MultivariateNormal, Uniform
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, ConstantLR
from pyknos.mdn.mdn import MultivariateGaussianMDN as mdn
from typing import Any, Callable, Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
from tqdm import tqdm
# from torchinfo import summary
from omegaconf import OmegaConf

from sbi.inference import SNPE_C
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.snpe.snpe_base import PosteriorEstimator
from sbi.utils import (
    check_dist_class,
    del_entries,
    test_posterior_net_for_multi_d_x,
    x_shape_from_simulation,
    del_entries,
)

from sbi.utils import (
    test_posterior_net_for_multi_d_x,
    x_shape_from_simulation,
    del_entries,
    validate_theta_and_x,
    handle_invalid_x,
    warn_if_zscoring_changes_data,
    nle_nre_apt_msg_on_invalid_x,
    npe_msg_on_invalid_x,
    mask_sims_from_prior,
)

import signal
import sys
sys.path.append('./src')

from dataset.Dataset_Classes import probR_HighD_Sets,Choice_Sampled_HighD_Dataset, Choice_Sampled_2D_Dataset, Data_Prefetcher
from dataset.Dataset_Classes import *

from utils.train import (
    plot_posterior_with_label,
    WarmupScheduler,
)
from utils.resource import(
    print_mem_info
)


DO_PRINT_MEM = False

def tensor_size(a):
    return a.element_size() * a.nelement() / (1024 ** 3)

def clean_cache():
    gc.collect()
    torch.cuda.empty_cache()
    
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MyPosteriorEstimator(PosteriorEstimator):
    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Union[str, Callable] = "maf",
        device: str = "gpu",
        logging_level: Union[int, str] = "INFO",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,    
    ):
        kwargs = del_entries(locals(), entries=("self", "__class__"))
        super().__init__(**kwargs)
    
    def train_base(
        self,
        config,
        prior_limits,
        dataloader_kwargs,
        continue_from_checkpoint=None,
        debug=False,
    ):
        """ train base model (first round for SNPE)
        """
        self.config = config
        self.log_dir = self.config.log_dir

        self._writer_hist = SummaryWriter(log_dir=f'{str(self.log_dir)}/event_hist')
        self._writer_fig  = SummaryWriter(log_dir=f'{str(self.log_dir)}/event_fig')

        self.prior_limits = prior_limits

        self.dataset_kwargs = self.config.dataset
        self.training_kwargs = self.config.train.training

        chosen_dur_trained_in_sequence  = self.dataset_kwargs.chosen_dur_trained_in_sequence
        use_data_prefetcher             = self.dataset_kwargs.use_data_prefetcher
        self.use_data_prefetcher        = use_data_prefetcher
        
        self.batch_counter = 0
        self.epoch_counter = 0
        self.dset_counter  = 0
        self.train_start_time = time.time()

        self.train_data_set_name_list = []
        self.val_data_set_name_list   = []

        self._summary["epoch_durations_sec"] = []
        self._summary["training_log_probs"] = []
        self._summary["validation_log_probs"] = []
        self._summary["learning_rates"] = []
        
        self._summary["best_from_epoch_of_current_dset"] = []
        self._summary["best_from_dset_of_current_dset"] = []
        self._summary["num_epochs_of_current_dset"] = []
        

        try:
            set_seed(config.seed)
            
            validation_fraction = OmegaConf.to_container(self.config.dataset.validation_fraction)
            val_set_names = self._fraction_2_set_names(validation_fraction)
            
            for self.run, chosen_dur in enumerate(chosen_dur_trained_in_sequence):
                
                # is_last_dset = False
                # initialization
                self._init_log_prob()
                self._best_model_from_epoch = -1
                
                val_loader,   x_val,   theta_val   = self._init_val(val_set_names, dataloader_kwargs, chosen_dur)
                train_loader, x_train, theta_train = self._init_train(val_set_names, dataloader_kwargs, chosen_dur)
                self._init_nn(x_train, theta_train, continue_from_checkpoint)

                self._collect_posterior_sets(x_train, theta_train, x_val, theta_val) # collect posterior sets before training

                train_start_time = time.time()
                training_kwargs = self.training_kwargs
                # train until loading new training dataset won't improve validation performance
                while (self.dset <= training_kwargs.max_num_dsets 
                       and not self._converged_dset()
                       ):
                    
                    # init optimizer and scheduler
                    self._init_optimizer(training_kwargs)

                    print(f'\n\n=== run {self.run}, chosen_dur {chosen_dur}, dset {self.dset} ===')
                    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    print_mem_info(f"\n{'gpu memory usage after loading dataset':46}", DO_PRINT_MEM)

                    self.epoch  = 0
                    
                    # train and validate until no validation performance improvement
                    while (
                        self.epoch <= training_kwargs.max_num_epochs
                        and not self._converged()
                        and (not debug or self.epoch <= 2)
                    ):
                        
                        # train and validate for one epoch
                        self.train_valid_one_epoch(
                            val_loader          = val_loader, 
                            train_loader        = train_loader, 
                            train_start_time    = train_start_time, 
                            training_kwargs     = training_kwargs,
                        )
                    
                    del train_loader, self.train_dataset
                    clean_cache()
                    
                    if debug:
                        break
                    
                    # load new training dataset for the next dset
                    self._describe_log_update_dset() # describe, log, update info of the training on the current dset
                    with torch.no_grad():
                        train_loader = self._get_train_loader(val_set_names, dataloader_kwargs, chosen_dur)
                        _, x_train, theta_train = self._get_fetcher_1st_batch_data(train_loader, len(train_loader))
                
                print(f"best val_log_prob: {self._best_val_log_prob:.2f}\n")
                if use_data_prefetcher:
                    train_prefetcher    = None
                    val_prefetcher      = None
                    del train_prefetcher, val_prefetcher
                    clean_cache()

                train_loader        = None
                val_loader          = None
                self.train_dataset  = None
                self.val_dataset    = None
                x                   = None
                theta               = None
                del train_loader, x, theta, self.val_dataset, self.train_dataset
                clean_cache()

            # Avoid keeping the gradients in the resulting network, which can cause memory leakage when benchmarking. save the network
            self._neural_net.zero_grad(set_to_none=True) # type: ignore
            torch.save(self._neural_net, os.path.join(self.config.log_dir, f"model/round_{self._round}_model.pt"))

            # save training curve
            self._plot_training_curve()

            return self, deepcopy(self._neural_net)

        finally:
            self.release_resources()

    def train_valid_one_epoch(self, val_loader, train_loader, train_start_time, training_kwargs, show_progress=True):
        
        with torch.no_grad():
            # fetcher same dataset for the coming epoch
            if self.use_data_prefetcher:
                start_time = time.time()
                print('updating epoch data-prefetcher ...', end=' ')
                                
                # reset prefetcher
                train_prefetcher = self._loader2prefetcher(train_loader)
                val_prefetcher   = self._loader2prefetcher(val_loader)
                                
                print(f'done in {(time.time()-start_time)/60:.2f} min')
                print_mem_info(f"\n\n{'gpu memory usage after prefetcher':46}", DO_PRINT_MEM)
                
            # plot the training curve
            if self.epoch > 0:
                self._plot_training_curve()

        # train and log one epoch
        self._neural_net.train()
        epoch_start_time, train_log_probs_sum = self._train_one_epoch(train_prefetcher if self.use_data_prefetcher else train_loader)
        train_log_prob_average = self._train_one_epoch_log(train_log_probs_sum)
        self._train_log_prob = train_log_prob_average
        
        # validate and log 
        self._neural_net.eval()
        with torch.no_grad():
            val_start_time = time.time()
                            
            # do validate and log
            self._val_log_prob = self._val_one_epoch(val_prefetcher if self.use_data_prefetcher else val_loader)
            self._val_one_epoch_log(train_start_time)
                            
            print(f"\nval_log_prob: {self._val_log_prob:.2f} in {(time.time() - val_start_time)/60:.2f} min")
            print_mem_info(f"{'gpu memory usage after validation':46}", DO_PRINT_MEM)

            # update epoch info and counter
            if show_progress:
                self._show_epoch_progress(self.epoch, epoch_start_time, train_log_prob_average, self._val_log_prob)
            self.epoch += 1
            self.epoch_counter += 1

        # update scheduler
        self._update_scheduler(training_kwargs)
        
    # ==================== Initialize functions ==================== #
    def _init_log_prob(self):
        
        # init log prob
        self.run    = 0
        self.dset   = 0
        self.epoch  = 0
        self._epoch_of_last_dset = 0
        self._val_log_prob, self._val_log_prob_dset = float("-Inf"), float("-Inf")
        self._best_val_log_prob, self._best_val_log_prob_dset = float("-Inf"), float("-Inf")
    
    def _init_val(self, val_set_names, dataloader_kwargs, chosen_dur):
        # prepare validation data
        use_data_prefetcher = self.use_data_prefetcher
        
        print('\npreparing [validation] data ...')
        val_loader = self._get_val_loader(val_set_names, dataloader_kwargs, chosen_dur)
        _, x_val, theta_val = self._get_fetcher_1st_batch_data(val_loader, len(val_loader))

        return val_loader, x_val, theta_val
    
    def _init_train(self, val_set_names, dataloader_kwargs, chosen_dur):
        # prepare training data
        print('\npreparing [training] data ...')
        train_loader = self._get_train_loader(val_set_names, dataloader_kwargs, chosen_dur)
        _, x, theta = self._get_fetcher_1st_batch_data(train_loader, len(train_loader))

        return train_loader, x, theta
        
    def _init_nn(self, x, theta, continue_from_checkpoint):
        
        # initialize neural net, move to device only once
        if self.run == 0 and self.dset_counter == 0:
            self._init_neural_net(x, theta, continue_from_checkpoint=continue_from_checkpoint)
         
    def _init_neural_net(self, x, theta, continue_from_checkpoint=None):
        
        if self._neural_net is None:
            
            # Use only training data for building the neural net (z-scoring transforms)
            self._neural_net = self._build_neural_net(
                theta[:3].to("cpu"),
                x[:3].to("cpu"),
            )
            self._x_shape = x_shape_from_simulation(x.to("cpu"))
            
            print('\nfinished build network')
            
            print(self._neural_net)
            
            test_posterior_net_for_multi_d_x(
                self._neural_net,
                theta.to("cpu"),
                x.to("cpu"),
            )
            
            if continue_from_checkpoint!=None and continue_from_checkpoint!='':
                print(f"loading neural net from '{continue_from_checkpoint}'")
                # load network from state dict 
                self._neural_net.load_state_dict(torch.load(continue_from_checkpoint))
            
        self._neural_net.to(self._device)
        
    def _init_optimizer(self, training_kwargs):
        warmup_epochs = self.config.train.training.warmup_epochs
        initial_lr    = self.config.train.training.initial_lr
        
        self.optimizer = optim.Adam(
                            list(self._neural_net.parameters()), 
                            lr=training_kwargs.learning_rate, 
                            weight_decay=eval(self.config.train.training.weight_decay) if isinstance(self.config.train.training.weight_decay, str) else self.config.train.training.weight_decay
                            )
        
        # warmup scheduler
        self.scheduler_warmup = WarmupScheduler(self.optimizer, warmup_epochs=warmup_epochs, init_lr=initial_lr, target_lr=training_kwargs['learning_rate'])
        
        # scheduler
        if training_kwargs['scheduler'] == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, **training_kwargs['scheduler_params'])
        if training_kwargs['scheduler'] == 'CosineAnnealingWarmRestarts':
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, **training_kwargs['scheduler_params'])
        if training_kwargs['scheduler'] == 'None': # constant lr
            self.scheduler = ConstantLR(self.optimizer, factor=1.0)
        # if training_kwargs['scheduler'] == 'CosineAnnealingLR':
        #     self.scheduler = CosineAnnealingLR(self.optimizer, *training_kwargs['scheduler_params'])       
    
    # ==================== dataloader / prefetcher ==================== #
    def _fraction_2_set_names(self, fraction):
        
        dataset_kwargs = self.config.dataset
        all_set_names = self._get_max_all_set_names(dataset_kwargs)
        
        # transfer fraction to set names
        if isinstance(fraction, float):
            num_sets   = int(fraction*len(all_set_names))
            set_names  = np.random.choice(all_set_names, num_sets, replace=False)
        elif isinstance(fraction, list):
            set_names  = [f'set_{i}' for i in fraction]
        else:
            raise ValueError('validation_fraction must be float or list')
        
        return set_names

    def _get_max_all_set_names(self, dataset_kwargs):
        
        # get all available set names
        data_path = self.config.data_path
        f = h5py.File(data_path, 'r', libver='latest', swmr=True)
        all_set_names = list(f.keys())
        num_total_sets = len(all_set_names)
        f.close()
        
        # select a subset of sets
        num_max_sets = dataset_kwargs['num_max_sets']
        num_max_sets = min(num_max_sets, num_total_sets)
        all_set_names = all_set_names[:num_max_sets]
        print(f'=== program seen {len(all_set_names)} sets from stored {num_total_sets} sets ===')
        
        return all_set_names
    
    def _get_dataset(self, set_names, num_chosen_theta_each_set, chosen_dur, theta_chosen_mode='random'):
        """Create the dataset given the set names"""
        
        if self.config.dataset.probR_sampling_place == 'collate_fn': # probR sampling in collate_fn
            dataset_class = probR_HighD_Sets if self.config.is_3_dim_dataset else probR_2D_Sets
        elif self.config.dataset.probR_sampling_place == 'in_dataset': # probR sampling in dataset
            dataset_class = Choice_Sampled_HighD_Dataset if self.config.is_3_dim_dataset else Choice_Sampled_2D_Dataset

        dataset = dataset_class(
            config = self.config,
            chosen_set_names = set_names,
            num_chosen_theta_each_set = num_chosen_theta_each_set,
            chosen_dur = chosen_dur,
            theta_chosen_mode = theta_chosen_mode,
            permutation_mode=self.config.dataset.permutation_mode,
        )
        
        return dataset

    def _get_loader(self, dataset, dataloader_kwargs, seed):
        """Create the dataloader given the dataset"""
        batch_size = self.config.dataset.batch_size
        num_examples = len(dataset)
        indices = torch.randperm(num_examples)
        
        loader_kwargs = {
            "batch_size": min(batch_size, num_examples),
            "drop_last" : True,
            "sampler"   : SubsetRandomSampler(indices.tolist()),
            "pin_memory": self.config.dataset.pin_memory,
        }
        if dataloader_kwargs is not None:
            loader_kwargs = dict(loader_kwargs, **dataloader_kwargs)

        print(f'loader_kwargs: {loader_kwargs}')
        
        g = torch.Generator()
        g.manual_seed(seed)
        
        return data.DataLoader(dataset, generator=g, **loader_kwargs)
    
    def _get_val_loader(self, val_set_names, dataloader_kwargs, chosen_dur):
        """ the val loader keeps the same for each dset, and each run
        """
        
        self.val_dataset = self._get_dataset(
            set_names = val_set_names,
            num_chosen_theta_each_set=self.config.dataset.validation_num_theta,
            chosen_dur=chosen_dur,
        )
        
        val_loader = self._get_loader(
            dataset=self.val_dataset,
            dataloader_kwargs=dataloader_kwargs,
            seed=self.config.seed, # no dset here
        )
        
        self.num_val_batches = len(val_loader)
        print(f'number of batches in the [val] dataset: {self.num_val_batches}')
        
        return val_loader
        
    def _get_train_loader(self, val_set_names, dataloader_kwargs, chosen_dur):
        """ the train loader updates each dset
        """
        dataset_kwargs = self.config.dataset
        all_set_names = self._get_max_all_set_names(dataset_kwargs) # get current all dataset names
        all_train_set_names = sorted(list(set(all_set_names) - set(val_set_names)))
        
        # first dset is trained with first element of the num_chosen_set list (e.g. 10 sets), 2nd -> 2nd ...
        num_train_sets = dataset_kwargs['num_train_sets'][self.dset%len(dataset_kwargs['num_train_sets'])]
        if self.dset>len(dataset_kwargs['num_train_sets'])-1:
            num_train_sets = dataset_kwargs['num_train_sets'][-1]
        self.num_train_sets = num_train_sets
        print(f'train_loader - num_train_sets: {num_train_sets}')
        
        assert num_train_sets <= len(all_train_set_names), 'not enough training sets'
        chosen_train_set_names = np.random.choice(all_train_set_names, num_train_sets, replace=False)
        print(f'containing: \n{chosen_train_set_names}')
        
        self.train_dataset = self._get_dataset(
            set_names = chosen_train_set_names,
            num_chosen_theta_each_set=self.config.dataset.num_chosen_theta_each_set,
            chosen_dur=chosen_dur,
        )
        
        train_loader = self._get_loader(
            dataset=self.train_dataset,
            dataloader_kwargs=dataloader_kwargs,
            seed=self.config.seed+self.dset,
        )
        
        print(f'number of batches in the [training] dataset: {len(train_loader)}')
        
        return train_loader
    
    def _loader2prefetcher(self, loader):
        prefetcher = Data_Prefetcher(loader, prefetch_factor=self.config.dataset.prefetch_factor)
        del loader
        clean_cache()
        return prefetcher
    
    
    def _get_fetcher_1st_batch_data(self, loader, num_batches):
        
        if use_data_prefetcher := self.config.dataset.use_data_prefetcher:
            prefetcher = self._loader2prefetcher(loader)
            x, theta = self._load_one_batch_data(prefetcher, use_data_prefetcher, num_batches)
        else:
            prefetcher = None
            x, theta = self._load_one_batch_data(loader, use_data_prefetcher, num_batches)

        return prefetcher, x, theta
    
    def _load_one_batch_data(self, train_prefetcher_or_loader, use_data_prefetcher, num_batches):
        
        print(f'loading 1 / {num_batches} batch of the dataset...', end=' ')
        start_time = time.time()
        
        if use_data_prefetcher:
            with torch.no_grad():
                x, theta = train_prefetcher_or_loader.next()
        else:
            x, theta = next(iter(train_prefetcher_or_loader)) # type: ignore
        
        print(f'takes {time.time() - start_time:.2f} seconds = {(time.time() - start_time) / 60:.2f} minutes')
        print('batch info of the dataset:',
                f'\n| x info  --> ', f'shape {x.shape}, dtype: {x.dtype}, device: {x.device}',
                f'\n| theta info  --> ', f'shape {theta.shape}, dtype: {theta.dtype}, device: {theta.device}',
                )
        
        return x, theta
    
    def _collect_posterior_sets(self, x, theta, x_val, theta_val):
        # load and show one example of the dataset
        
        print(f'\ncollect posterior sets...', end=' ')
        start_time = time.time()
        
        self.posterior_train_set = {
            'x'         : [],
            'x_shuffled': [],
            'theta'     : [],
        }
        
        self.posterior_val_set = {
            'x'         : [],
            'x_shuffled': [],
            'theta'     : [],
        }
        
        for i in range(self.config.train.posterior.val_set_size):
            self.posterior_train_set['x'].append(x[i, ...])
            self.posterior_train_set['x_shuffled'].append(x[i, ...][torch.randperm(x.shape[1])])
            self.posterior_train_set['theta'].append(theta[i, ...])
            
            self.posterior_val_set['x'].append(x_val[i, ...])
            self.posterior_val_set['x_shuffled'].append(x_val[i, ...][torch.randperm(x_val.shape[1])])
            self.posterior_val_set['theta'].append(theta_val[i, ...])
        # print(f'plotting ...', end=' ')
        # # collect of the dataset
        # for fig_idx in range(len(self.posterior_train_set['x'])):
            
        #     figure = plt.figure()
        #     plt.imshow(self.posterior_train_set['x'][fig_idx][:150, :].cpu())
        #     plt.savefig(f'{self.log_dir}/posterior/x_train_{fig_idx}_run{self.run}_dset{self.dset}.png')
        #     self._writer_fig.add_figure(f"data_run{self.run}/x_train_{fig_idx}", figure, self.dset)
        #     plt.close(figure)
            
        #     figure = plt.figure()
        #     plt.imshow(self.posterior_train_set['x_shuffled'][fig_idx][:150, :].cpu())
        #     plt.savefig(f'{self.log_dir}/posterior/x_train_{fig_idx}_run{self.run}_dset{self.dset}_shuffled.png')
        #     self._writer_fig.add_figure(f"data_run{self.run}/x_train_{fig_idx}_shuffled", figure, self.dset)
        #     plt.close(figure)
        
        
        #     figure = plt.figure()
        #     plt.imshow(self.posterior_val_set['x'][fig_idx][:150, :].cpu())
        #     plt.savefig(f'{self.log_dir}/posterior/x_train_{fig_idx}_run{self.run}_dset{self.dset}.png')
        #     self._writer_fig.add_figure(f"data_run{self.run}/x_val_{fig_idx}", figure, self.dset)
        #     plt.close(figure)
            
        #     figure = plt.figure()
        #     plt.imshow(self.posterior_val_set['x_shuffled'][fig_idx][:150, :].cpu())
        #     plt.savefig(f'{self.log_dir}/posterior/x_train_{fig_idx}_run{self.run}_dset{self.dset}_shuffled.png')
        #     self._writer_fig.add_figure(f"data_run{self.run}/x_val_{fig_idx}_shuffled", figure, self.dset)
        #     plt.close(figure)
            
            # self._summary_writer.flush()

        print(f'takes {time.time() - start_time:.2f} seconds = {(time.time() - start_time) / 60:.2f} minutes')
    
    # ==================== train / validation ==================== #
    def _train_one_epoch(self, train_prefetcher_or_loader, do_train=True):
        
        print_freq      = self.config.train.training.print_freq
        # clip_max_norm   = self.config.train.training.clip_max_norm
        
        epoch_start_time = time.time()
        batch_timer      = time.time()
        
        self.train_data_size    = 0
        train_log_probs_sum     = 0
        train_batch_num         = 0
        
        do_train = True
        # === train network === 
        if not do_train:
            train_loss = 0
            self.train_data_size = 1
            print('!! no training performed !!')
            
        if self.use_data_prefetcher:
            x, theta    = train_prefetcher_or_loader.next()
            
            while x is not None:
                self.optimizer.zero_grad()
                # train one batch and log progress
                # time_start = time.time()
                if do_train:
                    train_loss, train_log_probs_sum = self._train_one_batch(x, theta, train_log_probs_sum)
                else:
                    train_loss, train_log_probs_sum = 0, 0
                # print(f'train time: {(time.time() - time_start)*1000:.2f} ms')
                
                # time_start = time.time()
                self.optimizer.step()
                # print(f'optimizer time: {(time.time() - time_start)*1000:.2f} ms')
                
                # time_start = time.time()
                self._train_one_batch_log(train_prefetcher_or_loader, batch_timer, train_batch_num, train_loss)
                # print(f'log time: {(time.time() - time_start)*1000:.2f} ms')
                
                # get next batch
                train_batch_num += 1
                self.batch_counter += 1
                
                # time_start = time.time()
                with torch.no_grad():
                    x, theta = train_prefetcher_or_loader.next()
                # print(f'prefetcher time: {(time.time() - time_start)*1000:.2f} ms')
                if self.config.debug and train_batch_num>=3:
                    break
        else:
            # del x, theta
            # time_start = time.time()
            for x, theta in train_prefetcher_or_loader:
                self.optimizer.zero_grad()
                # print(f'loading time: {(time.time() - time_start)*1000:.2f} ms')    
                # time_start = time.time()
                # train one batch and log progress
                if do_train:
                    train_loss, train_log_probs_sum = self._train_one_batch(x, theta, train_log_probs_sum)
                else:
                    train_loss, train_log_probs_sum = 0, 0
                self.optimizer.step()
                # print(f'training time: {(time.time() - time_start)*1000:.2f} ms')
                # time_start = time.time()
                self._train_one_batch_log(train_prefetcher_or_loader, batch_timer, train_batch_num, train_loss)
                # print(f'loggin time: {(time.time() - time_start)*1000:.2f} ms')
                
                # get next batch
                train_batch_num += 1
                self.batch_counter += 1
                # time_start = time.time()
                if self.config.debug and train_batch_num>=3:
                    break
                
        return epoch_start_time,train_log_probs_sum

    def _train_one_epoch_log(self, train_log_probs_sum):
        
        train_log_prob_average = train_log_probs_sum / self.train_data_size
        
        # get current learning rate
        if self.epoch < self.config['train']['training']['warmup_epochs']:
            current_learning_rate  = self.scheduler_warmup.optimizer.param_groups[0]['lr']
        # elif self.config['train']['training']['scheduler'] != 'None':
        else:
            current_learning_rate  = self.scheduler.optimizer.param_groups[0]['lr']
            # current_learning_rate  = self.config['train']['training']['learning_rate']
        
        # train_log_prob_average = train_log_probs_sum / (self.num_train_batches * dataset_kwargs['batch_size'] * dataset_kwargs['num_probR_sample'])
        self._summary_writer.add_scalar("learning_rates", current_learning_rate, self.epoch_counter)
        self._summary_writer.add_scalars("log_probs", {'training': train_log_prob_average}, self.epoch_counter)
        
        # log the graident after each epoch
        # if self.config['train']['training']['log_gradients']:
        for name, param in self._neural_net.named_parameters():
            if param.requires_grad:
                self._writer_hist.add_histogram(f'Gradients/{name}', param.grad, self.epoch_counter)
        
        # log the bias, activations, layer, weights after each epoch
        # without loging the batch norm parameters
        # if self.config['train']['training']['log_weights']:
        for name, param in self._neural_net.named_parameters():
            if param.requires_grad:
                self._writer_hist.add_histogram(f'Weights/{name}', param, self.epoch_counter)
        
        # self._summary_writer.flush()
        
        self._summary["training_log_probs"].append(train_log_prob_average)
        self._summary["learning_rates"].append(current_learning_rate)
        
        return train_log_prob_average
    
    def _train_one_batch(self, x, theta, train_log_probs_sum):
        
        with torch.no_grad():
            x     = x.to(self._device)
            theta = theta.to(self._device)
            masks_batch = torch.ones_like(theta[:, 0]).to(self._device)

        # del x, theta
        self.train_data_size += len(x)
        
        # force_first_round_loss: If `True`, train with maximum likelihood,
        # i.e., potentially ignoring the correction for using a proposal
        # distribution different from the prior.
        train_losses = self._loss(
            theta,
            x,
            masks_batch,
            proposal=self._proposal_roundwise[-1],
            calibration_kernel = lambda x: ones([len(x)], device=self._device),
            force_first_round_loss=True,
        )
        print_mem_info('memory usage after loss', DO_PRINT_MEM)
        train_loss = torch.mean(train_losses)
        train_loss.backward()
        clean_cache()
        train_log_probs_sum -= train_losses.sum().item()

        clip_max_norm = self.config.train.training.clip_max_norm
        if clip_max_norm is not None:
            clip_grad_norm_(self._neural_net.parameters(), max_norm=clip_max_norm)
            
        del x, theta, masks_batch, train_losses
        gc.collect()
        torch.cuda.empty_cache()
        
        return train_loss, train_log_probs_sum 
    
    def _train_one_batch_log(self, train_prefetcher_or_loader, batch_timer, train_batch_num, train_loss, do_print_mem=DO_PRINT_MEM):
        
        print_freq = self.config.train.training.print_freq
        # print(len(train_prefetcher_or_loader), print_freq, len(train_prefetcher_or_loader)//print_freq, train_batch_num % (len(train_prefetcher_or_loader)//print_freq))
        if print_freq == 0: # do nothing
            pass

        elif len(train_prefetcher_or_loader) <= print_freq:
            # print_mem_info('memory usage after batch', DO_PRINT_MEM)
            print(f'epoch {self.epoch:4}: batch {train_batch_num:4}  train_loss {-1*train_loss:.2f}, time {(time.time() - batch_timer)/60:.2f}min')

        elif train_batch_num % (len(train_prefetcher_or_loader)//print_freq) == 0: # print every 5% of batches
            # print_mem_info('memory usage after batch', DO_PRINT_MEM)
            print(f'epoch {self.epoch:4}: batch {train_batch_num:4}  train_loss {-1*train_loss:.2f}, time {(time.time() - batch_timer)/60:.2f}min')
        
        
        self._summary_writer.add_scalar("train_loss_batch", train_loss, self.batch_counter)
        # self._summary_writer.flush()
    
    def _val_one_epoch(self, val_prefetcher_or_loader):
        
        val_data_size       = 0
        val_log_prob_sum    = 0
        
        if self.use_data_prefetcher: # use data prefetcher
            x_val, theta_val    = val_prefetcher_or_loader.next()
        
            while x_val is not None:
                
                x_val     = x_val.to(self._device)
                theta_val = theta_val.to(self._device)
                masks_batch = torch.ones_like(theta_val[:, 0])
                
                # del x_val, theta_val
                
                # update validation loss
                val_losses = self._loss(
                    theta_val,
                    x_val,
                    masks_batch,
                    proposal=self._proposal_roundwise[-1],
                    calibration_kernel = lambda x: ones([len(x)], device=self._device),
                    force_first_round_loss=True,
                )
                print_mem_info('\nmemory usage after loss', DO_PRINT_MEM)
                val_log_prob_sum -= val_losses.sum().item()
                val_data_size    += len(x_val)
                # print(f'{self.use_data_prefetcher} val_data_size {val_data_size}', end=' ')
                
                del x_val, theta_val, masks_batch, val_losses
                clean_cache()
                
                # get next batch
                x_val, theta_val = val_prefetcher_or_loader.next()
                # print_mem_info('\nmemory usage after batch', DO_PRINT_MEM)
                if self.config.debug:
                    break
                
        else: # use data loader
            for x_val, theta_val in val_prefetcher_or_loader:
                
                x_val       = x_val.to(self._device)
                theta_val   = theta_val.to(self._device)
                masks_batch = torch.ones_like(theta_val[:, 0])
                
                # update validation loss
                val_losses = self._loss(
                    theta_val,
                    x_val,
                    masks_batch,
                    proposal=self._proposal_roundwise[-1],
                    calibration_kernel = lambda x: ones([len(x)], device=self._device),
                    force_first_round_loss=True,
                )
                val_log_prob_sum -= val_losses.sum().item()
                val_data_size    += len(x_val)
                
                del x_val, theta_val, masks_batch
                clean_cache()
                
                if self.config.debug:
                    break
        
        return val_log_prob_sum / val_data_size
    
    def _val_one_epoch_log(self, train_start_time):
        
        # print(f'epoch {self.epoch}: val_log_prob {self._val_log_prob:.2f}')
        self._summary_writer.add_scalars("log_probs", {'validation': self._val_log_prob}, self.epoch_counter)
        # self._summary_writer.flush()
        
        self._summary["validation_log_probs"].append(self._val_log_prob)
        self._summary["epoch_durations_sec"].append(time.time() - train_start_time)
        # print('val logged')
    
    def _update_scheduler(self, training_kwargs):
        
        if self.epoch < self.config['train']['training']['warmup_epochs']:
            self.scheduler_warmup.step()
        elif training_kwargs['scheduler'] == 'ReduceLROnPlateau':
            self.scheduler.step(self._val_log_prob)
        # elif training_kwargs['scheduler'] == 'None':
        #     pass
        else:
            self.scheduler.step() # type: ignore
    
    # ==================== behavior plots ==================== #
    def _posterior_behavior_log(self, limits):
        
        config = self.config
        with torch.no_grad():
            
            epoch = self.epoch_counter-1
            current_net = deepcopy(self._neural_net)

            # if epoch%config['train']['posterior']['step'] == 0:
            posterior_start_time = time.time()
            print("--> Building posterior...", end=" ")

            posterior = self.build_posterior(current_net)
            self._model_bank = [] # clear model bank to avoid memory leak
            
            print(f"in {(time.time()-posterior_start_time)/60:.2f} min, Plotting ... ", end=" ")
            for fig_idx in range(len(self.posterior_train_set['x'])):
                print(f'{fig_idx}', end=' ')
                # plot posterior - train x
                fig_x, _ = plot_posterior_with_label(
                    posterior       = posterior, 
                    sample_num      = config.train.posterior.sampling_num,
                    x               = self.posterior_train_set['x'][fig_idx].to(self._device),
                    true_params     = self.posterior_train_set['theta'][fig_idx],
                    limits          = limits,
                    prior_labels    = config.prior.prior_labels,
                )
                plt.savefig(f"{self.log_dir}/posterior/figures/posterior_x_train_{fig_idx}_epoch_{epoch}.png")
                self._writer_fig.add_figure(f"posterior/x_train_{fig_idx}", fig_x, epoch)
                plt.close(fig_x)
                del fig_x, _
                clean_cache()
                
                # plot posterior - train x_shuffled
                fig_x, _ = plot_posterior_with_label(
                    posterior       = posterior, 
                    sample_num      = config.train.posterior.sampling_num,
                    x               = self.posterior_train_set['x_shuffled'][fig_idx].to(self._device),
                    true_params     = self.posterior_train_set['theta'][fig_idx],
                    limits          = limits,
                    prior_labels    = config.prior.prior_labels,
                )
                plt.savefig(f"{self.log_dir}/posterior/figures/posterior_x_train_{fig_idx}_epoch_{epoch}_shuffled.png")
                self._writer_fig.add_figure(f"posterior/x_train_{fig_idx}_shuffled", fig_x, epoch)
                plt.close(fig_x)
                del fig_x, _
                clean_cache()
                
                # plot posterior - val x
                fig_x_val, _ = plot_posterior_with_label(
                    posterior       = posterior, 
                    sample_num      = config.train.posterior.sampling_num,
                    x               = self.posterior_val_set['x'][fig_idx].to(self._device),
                    true_params     = self.posterior_val_set['theta'][fig_idx],
                    limits          = limits,
                    prior_labels    = config.prior.prior_labels,
                )
                plt.savefig(f"{self.log_dir}/posterior/figures/posterior_x_val_{fig_idx}_epoch_{epoch}.png")
                self._writer_fig.add_figure(f"posterior/x_val_{fig_idx}", fig_x_val, epoch)
                plt.close(fig_x_val)
                del fig_x_val, _
                clean_cache()
                
                # plot posterior - val x_shuffled
                fig_x_val, _ = plot_posterior_with_label(
                    posterior       = posterior, 
                    sample_num      = config.train.posterior.sampling_num,
                    x               = self.posterior_val_set['x_shuffled'][fig_idx].to(self._device),
                    true_params     = self.posterior_val_set['theta'][fig_idx],
                    limits          = limits,
                    prior_labels    = config.prior.prior_labels,
                )
                plt.savefig(f"{self.log_dir}/posterior/figures/posterior_x_val_{fig_idx}_epoch_{epoch}_shuffled.png")
                self._writer_fig.add_figure(f"posterior/x_val_{fig_idx}_shuffled", fig_x_val, epoch)
                plt.close(fig_x_val)
                del fig_x_val, _
                clean_cache()
                # self._summary_writer.flush()
                
            del posterior, current_net
            gc.collect()
            torch.cuda.empty_cache()
                
            print(f"finished in {(time.time()-posterior_start_time)/60:.2f}min")
            
    def _plot_training_curve(self):
        log_dir         = self.config.log_dir
        duration        = np.array(self._summary["epoch_durations_sec"])
        train_log_probs = self._summary["training_log_probs"]
        val_log_probs   = self._summary["validation_log_probs"]
        learning_rates  = self._summary["learning_rates"]
        best_val_log_prob = self._best_val_log_prob
        best_val_log_prob_epoch = self._best_model_from_epoch
        
        plt.tight_layout()
        
        fig, axes = plt.subplots(2,1, figsize=(16,10))
        fig.subplots_adjust(hspace=0.3)
        
        # plot learning rate
        ax0 = axes[0]
        ax0.plot(learning_rates, '-', label='lr', lw=2)
        # ax0.plot(best_val_log_prob_epoch, learning_rates[best_val_log_prob_epoch-1], 'v', color='tab:red', lw=2) # type: ignore

        ax0.set_xlabel('epochs')
        ax0.set_ylabel('learning rate')
        ax0.grid(alpha=0.2)
        ax0.set_title('training curve')

        ax1 = axes[1]
        ax1.plot(train_log_probs, '.-', label='training', alpha=0.8, lw=2, color='tab:blue', ms=0.1)
        ax1.plot(val_log_probs, '.-', label='validation', alpha=0.8, lw=2, color='tab:orange', ms=0.1)
        max_list = [max(val_log_probs), max(train_log_probs)]
        if ("test_log_probs" in self._summary.keys()) and (len(self._summary["test_log_probs"]) > 0):
            test_log_probs = self._summary["test_log_probs"]
            ax1.plot(test_log_probs, '.-', label='test', alpha=0.8, lw=2, color='tab:brown', ms=0.1)
            max_list.append(max(test_log_probs))
        
        # find best val log prob, and plot it
        best_val_log_prob = max(val_log_probs)
        best_val_log_prob_epoch = np.argmax(val_log_probs)
        
        ax1.plot(best_val_log_prob_epoch, best_val_log_prob, 'v', color='red', lw=2)
        ax1.text(best_val_log_prob_epoch, best_val_log_prob+0.02, f'{best_val_log_prob:.2f}', color='red', fontsize=10, ha='center', va='bottom') # type: ignore
        ax1.set_ylim(-25, max(max_list)+0.2)
        
        ax1.legend()
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('log_prob')
        ax1.grid(alpha=0.2)

        ax2 = ax1.twiny()
        ax2.plot((duration-duration[0])/60/60, max(val_log_probs)*np.ones_like(val_log_probs), '-', alpha=0)
        ax2.set_xlabel('time (hours)')

        # save the figure
        plt.savefig(f'{log_dir}/training_curve.png')
        print('saved training curve')
        plt.close()

    # ==================== converge and log ==================== #
    # def _converged(self):
    #     """Return whether the training converged yet and save best model state so far.

    #     Checks for improvement in validation performance over previous epochs.

    #     Args:
    #         epoch: Current epoch in training.
    #         stop_after_epochs: How many fruitless epochs to let pass before stopping.

    #     Returns:
    #         Whether the training has stopped improving, i.e. has converged.
    #     """
    #     epoch                   = self.epoch
    #     improvement_threshold   = self.config.train.training.improvement_threshold
    #     min_num_epochs          = self.config.train.training.min_num_epochs
    #     stop_after_epochs       = self.config.train.training.stop_after_epochs
        
    #     converged = False

    #     assert self._neural_net is not None
    #     neural_net = self._neural_net

    #     # (Re)-start the epoch count with the first epoch or any improvement. 
    #     improvement = self._val_log_prob - self._best_val_log_prob
    #     if epoch == 0 or ((self._val_log_prob > self._best_val_log_prob) and (improvement >= improvement_threshold)):
            
    #         self._epochs_since_last_improvement = 0
            
    #         self._best_val_log_prob     = self._val_log_prob
    #         self._best_model_state_dict = deepcopy(neural_net.state_dict())
    #         self._best_model_from_epoch = epoch - 1 # TODO update converge conditions
            
    #         if epoch != 0: #and epoch%self.config['train']['posterior']['step'] == 0:
    #             self._posterior_behavior_log(self.prior_limits) # plot posterior behavior when best model is updated
    #             print_mem_info(f"{'gpu memory usage after posterior behavior log':46}", DO_PRINT_MEM)
    #         # torch.save(deepcopy(neural_net.state_dict()), f"{self.log_dir}/model/best_model_state_dict_run{self.run}.pt")
            
    #     else:
    #         self._epochs_since_last_improvement += 1

    #     # If no validation improvement over many epochs, stop training.
    #     if self._epochs_since_last_improvement > stop_after_epochs - 1 and epoch > self._epoch_of_last_dset+min_num_epochs:
    #         # neural_net.load_state_dict(self._best_model_state_dict)
    #         converged = True
            
    #         self._neural_net.load_state_dict(self._best_model_state_dict)
    #         self._val_log_prob = self._best_val_log_prob
    #         self._epochs_since_last_improvement = 0
    #         # self._epoch_of_last_dset = epoch - 1
    #         self._epoch_of_last_dset = self.epoch_counter-1
        
    #     # log info for this dset
    #     self._summary_writer.add_scalar(f"run{self.run}/best_val_epoch_glob", self._best_model_from_epoch, self.epoch_counter-1)
    #     self._summary_writer.add_scalar(f"run{self.run}/best_val_log_prob_glob", self._best_val_log_prob, self.epoch_counter-1)
    #     self._summary_writer.add_scalar(f"run{self.run}/current_dset_glob", self.dset_counter, self.epoch_counter-1)
    #     self._summary_writer.add_scalar(f"run{self.run}/num_chosen_dset_glob", self.num_train_sets, self.epoch_counter-1)
    #     self._summary_writer.flush()
    #     return converged
    
    def _converged(self):
        """Return whether the training converged yet and save best model state so far.

        Checks for improvement in validation performance over previous epochs.

        Args:
            epoch: Current epoch in training.
            stop_after_epochs: How many fruitless epochs to let pass before stopping.

        Returns:
            Whether the training has stopped improving, i.e. has converged.
        """
        epoch                   = self.epoch
        improvement_threshold   = self.config.train.training.improvement_threshold
        min_num_epochs          = self.config.train.training.min_num_epochs
        stop_after_epochs       = self.config.train.training.stop_after_epochs
        
        converged = False

        assert self._neural_net is not None
        neural_net = self._neural_net

        # (Re)-start the epoch count with the first epoch or any improvement. 
        improvement = self._val_log_prob - self._best_val_log_prob
        if (epoch == 0 
            or (
                (self._val_log_prob > self._best_val_log_prob) 
                and (improvement >= improvement_threshold)
                )
            ):
            
            self._epochs_since_last_improvement = 0
            
            self._best_val_log_prob     = self._val_log_prob
            self._best_model_state_dict = deepcopy(neural_net.state_dict())
            self._best_model_from_epoch = epoch - 1
            
            if epoch != 0: #and epoch%self.config['train']['posterior']['step'] == 0:
                if self.config.train.posterior.plot_posterior and epoch%self.config.train.posterior.step == 0:
                    self._posterior_behavior_log(self.prior_limits) # plot posterior behavior when best model is updated
                print_mem_info(f"{'gpu memory usage after posterior behavior log':46}", DO_PRINT_MEM)
            # torch.save(deepcopy(neural_net.state_dict()), f"{self.log_dir}/model/best_model_state_dict_run{self.run}.pt")
            
        else:
            self._epochs_since_last_improvement += 1

        # If no validation improvement over many epochs, stop training.
        if self._epochs_since_last_improvement > stop_after_epochs - 1 and epoch > min_num_epochs:
            # neural_net.load_state_dict(self._best_model_state_dict)
            converged = True
            
            self._neural_net.load_state_dict(self._best_model_state_dict)
            self._val_log_prob = self._best_val_log_prob
            self._epochs_since_last_improvement = 0
            # self._epoch_of_last_dset = epoch - 1
            self._epoch_of_last_dset = self.epoch_counter-1
        
        # log info for this dset
        self._summary_writer.add_scalar(f" (epoch) run{self.run}/best_val_epoch_glob", self._best_model_from_epoch, self.epoch_counter-1)
        self._summary_writer.add_scalar(f" (epoch) run{self.run}/best_val_log_prob_glob", self._best_val_log_prob, self.epoch_counter-1)
        self._summary_writer.add_scalar(f" (epoch) run{self.run}/current_dset_glob", self.dset_counter, self.epoch_counter-1)
        # self._summary_writer.add_scalar(f" (epoch) run{self.run}/num_chosen_dset_glob", self.num_train_sets, self.epoch_counter-1)
        self._summary_writer.flush()
        return converged
    
    # def _converged_dset(self):
        
    #     improvement_threshold = self.config.train.training.improvement_threshold
    #     min_num_dsets         = self.config.train.training.min_num_dsets
    #     stop_after_dsets      = self.config.train.training.stop_after_dsets
        
    #     converged = False
    #     assert self._neural_net is not None
        
    #     # improvement = self._val_log_prob - self._best_val_log_prob
    #     if self.dset == 0 or (self._val_log_prob_dset > self._best_val_log_prob_dset):
            
    #         self._dset_since_last_improvement = 0
            
    #         self._best_val_log_prob_dset        = self._val_log_prob_dset
    #         self._best_model_state_dict_dset    = deepcopy(self._neural_net.state_dict())
    #         self._best_model_from_dset          = self.dset - 1
            
    #         torch.save(deepcopy(self._neural_net.state_dict()), f"{self.log_dir}/model/best_model_state_dict_run{self.run}.pt")
        
    #     else:
    #         self._dset_since_last_improvement += 1
            
    #     if self._dset_since_last_improvement > stop_after_dsets - 1 and self.dset > min_num_dsets - 1:
            
    #         converged = True
            
    #         self._neural_net.load_state_dict(self._best_model_state_dict_dset)
    #         self._val_log_prob_dset = self._best_val_log_prob_dset
    #         self._dset_since_last_improvement = 0
            
    #         torch.save(deepcopy(self._neural_net.state_dict()), f"{self.log_dir}/model/best_model_state_dict_run{self.run}.pt")
        
    #     # use only the whole dataset as the training set, would train only once
    #     if self.config.dataset.one_dataset == True and self.dset == 1:
    #         converged = True
        
    #     return converged
    
    def _converged_dset(self):
        
        improvement_threshold = self.config.train.training.improvement_threshold
        min_num_dsets         = self.config.train.training.min_num_dsets
        stop_after_dsets      = self.config.train.training.stop_after_dsets
        
        converged = False
        assert self._neural_net is not None
        
        # improvement = self._val_log_prob - self._best_val_log_prob
        self._summary["num_epochs_of_current_dset"].append(self._epoch_of_last_dset)
        if self.dset == 0 or (self._val_log_prob_dset > self._best_val_log_prob_dset):
            
            self._dset_since_last_improvement = 0
            
            self._best_val_log_prob_dset        = self._val_log_prob_dset
            self._best_model_state_dict_dset    = deepcopy(self._neural_net.state_dict())
            self._best_model_from_dset          = self.dset - 1
            
            torch.save(deepcopy(self._neural_net.state_dict()), f"{self.log_dir}/model/best_model_state_dict_run{self.run}.pt")

            self._summary["best_from_epoch_of_current_dset"].append(self._best_model_from_epoch)
            self._summary["best_from_dset_of_current_dset"].append(self._best_model_from_dset)
            
        else:
            self._dset_since_last_improvement += 1
            
        if self._dset_since_last_improvement > stop_after_dsets - 1 and self.dset > min_num_dsets - 1:
            
            converged = True
            
            self._neural_net.load_state_dict(self._best_model_state_dict_dset)
            self._val_log_prob_dset = self._best_val_log_prob_dset
            self._dset_since_last_improvement = 0
            
            torch.save(deepcopy(self._neural_net.state_dict()), f"{self.log_dir}/model/best_model_state_dict_run{self.run}.pt")
        
        # use only the whole dataset as the training set, would train only once
        # check if one_dataset is in self.config.dataset
        if hasattr(self.config.dataset, "one_dataset"):
            if self.config.dataset.one_dataset == True and self.dset == 1:
                converged = True
        else:
            print("no one_dataset in config.dataset")
        
        self._summary_writer.add_scalar(" (dset) num_epochs_of_current_dset", self._epoch_of_last_dset, self.dset_counter-1)
        self._summary_writer.add_scalar(" (dset) best_from_epoch_of_current_dset", self._best_model_from_epoch, self.dset_counter-1)
        self._summary_writer.add_scalar(" (dset) best_from_dset_of_current_dset", self._best_model_from_dset, self.dset_counter-1)
        
        best_idxs = self._get_best_epoch_idx()
        self._summary_writer.add_text(" (dset) best_epoches_till_now", best_idxs, self.dset_counter-1)
        
        return converged
    
    def _get_best_epoch_idx(self):
        """compute the best epoch index trace till now
        check converge_test.py for more details
        """
        
        best_dset_idx = np.array(self._summary["best_from_dset_of_current_dset"][1:], dtype=int)
        starting_epoch_dset = np.array(self._summary["num_epochs_of_current_dset"][1:-1], dtype=int)+1
        starting_epoch_dset = np.insert(starting_epoch_dset, 0, 0)
        best_epoch = np.array(self._summary["best_from_epoch_of_current_dset"][1:], dtype=int)
        
        best_idxs = [starting_epoch_dset[dset] + best_epoch for dset, best_epoch in zip( best_dset_idx, best_epoch)]
        self._summary["best_epoches"] = best_idxs
        best_idxs = str(list(best_idxs))
        print("best_epochs so far", best_idxs)
        
        return best_idxs
        
        
    def _show_epoch_progress(self, epoch, starting_time, train_log_prob, val_log_prob):
        print(f"| Epochs trained: {epoch:4} | log_prob train: {train_log_prob:.2f} | log_prob val: {val_log_prob:.2f} | . Time elapsed {(time.time()-starting_time)/ 60:6.2f}min, trained in total {(time.time() - self.train_start_time)/60:6.2f}min")
    
    def _show_epoch_progress_with_test(self, epoch, starting_time, train_log_prob, val_log_prob, test_log_prob):
        print(f"| Epochs trained: {epoch-1:4} | log_prob train: {train_log_prob:.2f} | log_prob val: {val_log_prob:.2f} | log_prob test: {test_log_prob:.2f} | . Time elapsed {(time.time()-starting_time)/ 60:6.2f}min, trained for {(time.time() - self.train_start_time)/60:6.2f}min")
    
    def _describe_log_update_dset(self):
        
        info = f"""
        -------------------------
        ||||| RUN {self.run} dset {self.dset} STATS |||||:
        -------------------------
        Total epochs trained: {self.epoch_counter}
        Best validation performance: {self._best_val_log_prob:.4f}, from epoch {self._best_model_from_epoch:5}
        Model from best epoch {self._best_model_from_epoch} is loaded for further training
        -------------------------
        """
        print(info)
        # log info for this dset
        self._summary_writer.add_scalar(f"run{self.run}/best_val_epoch_of_dset", self._best_model_from_epoch, self.epoch_counter-1)
        self._summary_writer.add_scalar(f"run{self.run}/best_val_log_prob", self._best_val_log_prob, self.epoch_counter-1)
        # self._summary_writer.flush()
        # self._summary_writer.add_scalar(f"run{self.run}/current_dset", self.dset_counter, self.epoch_counter)
        # self._summary_writer.add_scalar(f"run{self.run}/num_chosen_dset", self.num_train_sets, self.epoch_counter)
        
        # update dset info
        self._val_log_prob_dset = self._best_val_log_prob
        
        # load data for next dset
        self.dset         += 1
        self.dset_counter += 1
    
    # ==================== release resources ==================== #
    def release_resources(self):
        # Release GPU resources
        train_loader        = None
        train_prefetcher    = None
        val_loader          = None
        val_prefetcher      = None
        self.empty_mem('cuda cache emptied')
        # release cpu resources by force
        self._neural_net.cpu()
        self._neural_net = None
        self.empty_mem('cpu cache emptied')
        # clear self
        self._do_clear()
        print('self cleared')

    def empty_mem(self, arg0):
        gc.collect()
        torch.cuda.empty_cache()
        print(arg0)

    def _clear_loaders(self, train_loader):
        
        del train_loader, self.train_dataset
        gc.collect()
        torch.cuda.empty_cache()
        
    def _do_clear(self):
        
        self._neural_net            = None  
        self._val_log_prob          = None  
        self._summary               = None  
        self._summary_writer        = None  
        self._round                 = None 
        self.train_dataset          = None
        self.val_dataset            = None
        
        self._best_val_log_prob     = None  
        self._val_log_prob_dset     = None 
        self._best_val_log_prob_dset= None
        

class MyPosteriorEstimator_P3(MyPosteriorEstimator):
    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Union[str, Callable] = "maf",
        device: str = "gpu",
        logging_level: Union[int, str] = "INFO",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,    
    ):
        kwargs = del_entries(locals(), entries=("self", "__class__"))
        super().__init__(**kwargs)
    
    def train_base_p3(
        self,
        config,
        prior_limits,
        dataloader_kwargs,
        continue_from_checkpoint=None,
        debug=False,
    ):
        self.config         = config
        self.log_dir        = config.log_dir
        self.prior_limits   = prior_limits

        self.dataset_kwargs = self.config.dataset
        self.training_kwargs = self.config.train.training
        
        self._writer_hist = SummaryWriter(log_dir=f'{str(self.log_dir)}/event_hist')
        self._writer_fig  = SummaryWriter(log_dir=f'{str(self.log_dir)}/event_fig')
        
        chosen_dur_trained_in_sequence  = self.dataset_kwargs.chosen_dur_trained_in_sequence
        use_data_prefetcher             = self.dataset_kwargs.use_data_prefetcher
        self.use_data_prefetcher        = use_data_prefetcher
        
        self.batch_counter = 0
        self.epoch_counter = 0
        self.dset_counter  = 0
        self.train_start_time = time.time()

        self.train_data_set_name_list = []
        self.val_data_set_name_list   = []
        
        self._summary["epoch_durations_sec"] = []
        self._summary["training_log_probs"] = []
        self._summary["validation_log_probs"] = []
        self._summary["test_log_probs"] = []
        self._summary["learning_rates"] = []
        
        self._summary["best_from_epoch_of_current_dset"] = []
        self._summary["best_from_dset_of_current_dset"] = []
        self._summary["num_epochs_of_current_dset"] = []
        
        set_seed(config.seed)
        
        chosen_dur = chosen_dur_trained_in_sequence[0]
        
        # ==================== initialize ==================== #
        # initialize log probs and counters
        self._init_log_prob()
        self._best_model_from_epoch = -1
        
        # initialize test dataset and test loader -> self.test_dataset, self.test_loader
        test_fraction = OmegaConf.to_container(self.config.dataset.test_fraction)
        test_set_names = self._fraction_2_set_names(test_fraction)
        x_test, theta_test = self._init_test_set(test_set_names, dataloader_kwargs, chosen_dur)
        
        # initialize and build the network
        self._init_neural_net(x_test, theta_test, continue_from_checkpoint)
        
        # collect posterior sets before training
        x_train, theta_train, _, _ = self._init_train_val_set(test_set_names, dataloader_kwargs, chosen_dur)
        self._collect_posterior_sets(x_train, theta_train, x_test, theta_test) #TODO: delete train, val, test data
        
        # set a train set name loader for loading train set names in different dsets
        train_set_name_loader = self._load_set_names()
        # ==================== training ==================== #
        train_start_time = time.time()
        while (self.dset <= self.training_kwargs.max_num_dsets 
                and not self._converged_dset() # TODO: check dset condition
                ):
            
            print(f'\n\n=== run {self.run}, chosen_dur {chosen_dur}, dset {self.dset} ===')
            tic = time.time()
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print_mem_info(f"\n{'gpu memory usage after loading dataset':46}", DO_PRINT_MEM)
            
            # load new train set names
            train_set_names = next(train_set_name_loader)
            print(f'\n===== train_set_names: {train_set_names} =====\n')
            if len(train_set_names) == 0:
                break
            
            # get train and val loaders for the current dset
            train_loader, val_loader = self._get_train_val_loaders(
                train_set_names = train_set_names,
                dataloader_kwargs = dataloader_kwargs,
                chosen_dur = chosen_dur,
            )
            
            # init optimizer and scheduler
            self._init_optimizer(self.training_kwargs)
            self.epoch  = 0
            
            print(f'\npreparing dataset takes {(time.time()-tic)/60:.2f} min')
            
            # train and validate until no validation performance improvement
            while (
                self.epoch <= self.training_kwargs.max_num_epochs
                and not self._converged()
                and (not debug or self.epoch <= 1)
            ):  
                epoch_start_time = time.time()
                check_test = self.config.dataset.check_test_perf
                
                # train and validate for one epoch
                self.train_valid_one_epoch(
                    val_loader          = val_loader, 
                    train_loader        = train_loader, 
                    train_start_time    = train_start_time, 
                    training_kwargs     = self.training_kwargs,
                    show_progress       = not check_test,
                )
                
                # test for one epoch
                if check_test:
                    with torch.no_grad():
                        tic = time.time()
                        print('testing one epoch ...', end=' ')
                        if self.use_data_prefetcher:
                            test_prefetcher = self._loader2prefetcher(self.test_loader)
                        self._test_log_prob = self._val_one_epoch( test_prefetcher if self.use_data_prefetcher else self.test_loader )
                        print(f'in {time.time()-tic:.2f} sec -->', end=' ')
                        self._test_one_epoch_log()
                        self._show_epoch_progress_with_test(self.epoch, epoch_start_time, self._train_log_prob, self._val_log_prob, self._test_log_prob)
            
            with torch.no_grad():
                # clear train loader, prefetcher, and val_prefetcher
                del train_loader, val_loader
                clean_cache()
                
                # describe, log, update info of the training on the current dset # TODO: log info check
                self._describe_log_update_dset()
                
        # clear
        print(f"best val_log_prob: {self._best_val_log_prob:.2f}")
        # del self.val_dataset, self.train_dataset
        # if use_data_prefetcher:
            # del train_prefetcher, val_prefetcher
        clean_cache()

        # Avoid keeping the gradients in the resulting network, which can cause memory leakage when benchmarking. save the network
        self._neural_net.zero_grad(set_to_none=True) # type: ignore
        torch.save(self._neural_net, os.path.join(self.config.log_dir, f"model/round_{self._round}_model.pt"))

        # save training curve
        self._plot_training_curve()

        return self, deepcopy(self._neural_net)
                            
    def _init_test_set(self, test_set_names, dataloader_kwargs, chosen_dur):
        """initialize test dataset and loader"""
        
        print('\n=== test dataset ===')
        # get test dataset
        self.test_dataset = self._get_dataset(
            set_names = test_set_names,
            num_chosen_theta_each_set=self.config.dataset.test_num_theta,
            chosen_dur=chosen_dur,
        )
        print_mem_info('after loading test_dataset', DO_PRINT_MEM)
        
        # get test loader
        print('\n=== test loader ===')
        self.test_loader = self._get_loader(
            dataset=self.test_dataset,
            dataloader_kwargs=dataloader_kwargs,
            seed=self.config.seed
        )
        print_mem_info('after loading test_loader', DO_PRINT_MEM)
        
        # get 1st batch data
        x_test, theta_test = next(iter(self.test_loader))
        # _, x_test, theta_test = self._get_fetcher_1st_batch_data(self.test_loader, len(self.test_loader))
        # del _
        # clean_cache()
        
        return x_test, theta_test
    
    def _init_train_val_set(self, test_set_names, dataloader_kwargs, chosen_dur):
        """initialize train dataset and loader"""
        
        print('\npreparing [training / validation] data ...')
        
        # get all training dataset names
        self.all_set_names = self._get_max_all_set_names(self.config.dataset) # get current all dataset names
        self.all_train_set_names = sorted(list(set(self.all_set_names) - set(test_set_names)))
        
        # get validation and training fraction
        self.validation_fraction = int(self.config.dataset.validation_fraction*100)
        self.train_fraction = int((1-self.config.dataset.validation_fraction)*100)
        
        # define train set name (choosen the first set)
        # train_set_name = [self.all_train_set_names[i] for i in range(self.config.dataset.increment_params['init'])]
        train_set_name = [self.all_train_set_names[0]]
        
        # get train and validation loaders for the first init set
        train_loader, val_loader = self._get_train_val_loaders(train_set_name, dataloader_kwargs, chosen_dur)
        
        # get 1st batch data
        print('\ngetting 1st [training] batch data ... ', end='')
        tic = time.time()
        x_train, theta_train = next(iter(train_loader))
        # _, x_train, theta_train = self._get_fetcher_1st_batch_data(train_loader, len(train_loader))
        print(f'in {time.time()-tic:.2f} sec')
        
        print('\ngetting 1st [val] batch data ... ', end='')
        tic = time.time()
        x_val, theta_val = next(iter(val_loader))
        # _, x_val, theta_val = self._get_fetcher_1st_batch_data(val_loader, len(val_loader))
        print(f'in {time.time()-tic:.2f} sec')
        
        del train_loader, val_loader
        clean_cache()
        return x_train, theta_train, x_val, theta_val
    
    def _get_train_val_loaders(self, train_set_names, dataloader_kwargs, chosen_dur):
        
        """get the train and validation loaders given the train_set_names"""
        
        print('=== training dataset ===')
        # if self.train_dataset exists, delete it
        if hasattr(self, 'train_dataset'):
            del self.train_dataset
            del self.val_dataset
            clean_cache()
            
        self.train_dataset = self._get_dataset(
            set_names = train_set_names,
            num_chosen_theta_each_set=self.config.dataset.train_num_theta,
            chosen_dur=chosen_dur,
            theta_chosen_mode = f'first_{self.train_fraction}', # 'first_90'
        )
        print_mem_info('after loading train_dataset', DO_PRINT_MEM)
        
        print()
        print('=== validation dataset ===')
        self.val_dataset = self._get_dataset(
            set_names = train_set_names,
            num_chosen_theta_each_set=self.config.dataset.train_num_theta,
            chosen_dur=chosen_dur,
            theta_chosen_mode=f'last_{self.validation_fraction}', # 'last_10'
        )
        print_mem_info('after loading val_dataset', DO_PRINT_MEM)
        
        print('\n=== training loader === ', end='')
        train_loader = self._get_loader(
            dataset = self.train_dataset,
            dataloader_kwargs=dataloader_kwargs,
            seed=self.config.seed+self.dset,
        )
        print_mem_info('after loading training_loader', DO_PRINT_MEM)
        print(f'{len(train_loader)} batches')
        
        print('\n=== validation loader ===', end='')
        val_loader = self._get_loader(
            dataset = self.val_dataset,
            dataloader_kwargs=dataloader_kwargs,
            seed=self.config.seed+self.dset,
        )
        print_mem_info('after loading val_loader', DO_PRINT_MEM)
        print(f'{len(val_loader)} batches')
        
        return train_loader, val_loader
    
    def _load_set_names(self):
        """
        Function to get sets to load given the current step and maximum load size.

        dset (int): Current dataset step.
        step (int): Increment step for each dset. Default is 2.
        max_load (int): Maximum number of sets to load. Default is 10.
        loop (bool): Whether to start over from the beginning when reaching the end of the dataset.

        Returns:
            list: List of set names to load.
        """
        init_num    = self.dataset_kwargs.increment_params['init']
        step        = self.dataset_kwargs.increment_params['step']
        max_load    = self.dataset_kwargs.increment_params['max_load']
        only_one_set= self.dataset_kwargs.increment_params['only_one_set']

        all_sets = self.all_train_set_names
        sets_in_queue = all_sets.copy()
        loaded_sets = deque(maxlen=max_load)
        
        # initialize loaded_sets with init_num sets
        for _ in range(init_num):
            loaded_sets.append(sets_in_queue.pop(0))
        yield list(loaded_sets)

        while True:
            # when only_one_set is True, only load one set in the whole training process
            if only_one_set:
                yield []
                
            for _ in range(step):
                if len(sets_in_queue) == 0:
                    sets_in_queue = all_sets.copy()
                    self.run += 1
                loaded_sets.append(sets_in_queue.pop(0))

            yield list(loaded_sets)
            
    def _test_one_epoch_log(self):
        
        self._summary_writer.add_scalars("log_probs", {'test': self._test_log_prob}, self.epoch_counter-1)
        self._summary["test_log_probs"].append(self._test_log_prob)
        print(f"test log prob: {self._test_log_prob:.4f}")
      
    
class MySNPE_C(SNPE_C, MyPosteriorEstimator):
    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Union[str, Callable] = "maf",
        device: str = "gpu",
        logging_level: Union[int, str] = "INFO",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,    
    ):
        kwargs = del_entries(locals(), entries=("self", "__class__"))
        # print(kwargs)
        super().__init__(**kwargs)
    
    def train(
        self,
        config,
        prior_limits,
        dataloader_kwargs,
        continue_from_checkpoint=None,
        debug=False,
    ) -> nn.Module:
        r"""Return density estimator that approximates the distribution $p(\theta|x)$.

        Args:
            num_atoms: Number of atoms to use for classification.
            training_batch_size: Training batch size.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            stop_after_epochs: The number of epochs to wait for improvement on the
                validation set before terminating training.
            max_num_epochs: Maximum number of epochs to run. If reached, we stop
                training even when the validation loss is still decreasing. Otherwise,
                we train until validation loss increases (see also `stop_after_epochs`).
            clip_max_norm: Value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.
            calibration_kernel: A function to calibrate the loss with respect to the
                simulations `x`. See Lueckmann, Gonçalves et al., NeurIPS 2017.
            resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If `True`, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time `.train()` was called.
            force_first_round_loss: If `True`, train with maximum likelihood,
                i.e., potentially ignoring the correction for using a proposal
                distribution different from the prior.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less targeted
                samples.
            use_combined_loss: Whether to train the neural net also on prior samples
                using maximum likelihood in addition to training it on all samples using
                atomic loss. The extra MLE loss helps prevent density leaking with
                bounded priors.
            retrain_from_scratch: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.
            show_train_summary: Whether to print the number of epochs and validation
                loss and leakage after the training.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn)

        Returns:
            Density estimator that approximates the distribution $p(\theta|x)$.
        """

        # WARNING: sneaky trick ahead. We proxy the parent's `train` here,
        # requiring the signature to have `num_atoms`, save it for use below, and
        # continue. It's sneaky because we are using the object (self) as a namespace
        # to pass arguments between functions, and that's implicit state management.
        self.config = config
        self._num_atoms = self.config.train.training.num_atoms
        self._use_combined_loss = self.config.train.training.use_combined_loss
        kwargs = del_entries(
            locals(), entries=("self", "__class__", "num_atoms", "use_combined_loss")
        )

        self._round = max(self._data_round_index)

        if self._round > 0:
            # Set the proposal to the last proposal that was passed by the user. For
            # atomic SNPE, it does not matter what the proposal is. For non-atomic
            # SNPE, we only use the latest data that was passed, i.e. the one from the
            # last proposal.
            proposal = self._proposal_roundwise[-1]
            self.use_non_atomic_loss = (
                isinstance(proposal, DirectPosterior)
                and isinstance(proposal.posterior_estimator._distribution, mdn)
                and self._neural_net is not None
                and isinstance(self._neural_net._distribution, mdn)
                and check_dist_class(
                    self._prior, class_to_check=(Uniform, MultivariateNormal)
                )[0]
            )

            algorithm = "non-atomic" if self.use_non_atomic_loss else "atomic"
            print(f"Using SNPE-C with {algorithm} loss")

            if self.use_non_atomic_loss:
                # Take care of z-scoring, pre-compute and store prior terms.
                self._set_state_for_mog_proposal()

        return super().train_base(**kwargs) # type: ignore
    

class MySNPE_C_P3(SNPE_C, MyPosteriorEstimator_P3):
    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Union[str, Callable] = "maf",
        device: str = "gpu",
        logging_level: Union[int, str] = "INFO",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,    
    ):
        kwargs = del_entries(locals(), entries=("self", "__class__"))
        # print(kwargs)
        super().__init__(**kwargs)
    
    def train(
        self,
        config,
        prior_limits,
        dataloader_kwargs,
        continue_from_checkpoint=None,
        debug=False,
    ) -> nn.Module:
        
        self.config = config
        self._num_atoms = self.config.train.training.num_atoms
        self._use_combined_loss = self.config.train.training.use_combined_loss
        kwargs = del_entries(
            locals(), entries=("self", "__class__", "num_atoms", "use_combined_loss")
        )

        self._round = max(self._data_round_index)

        if self._round > 0:
            # Set the proposal to the last proposal that was passed by the user. For
            # atomic SNPE, it does not matter what the proposal is. For non-atomic
            # SNPE, we only use the latest data that was passed, i.e. the one from the
            # last proposal.
            proposal = self._proposal_roundwise[-1]
            self.use_non_atomic_loss = (
                isinstance(proposal, DirectPosterior)
                and isinstance(proposal.posterior_estimator._distribution, mdn)
                and self._neural_net is not None
                and isinstance(self._neural_net._distribution, mdn)
                and check_dist_class(
                    self._prior, class_to_check=(Uniform, MultivariateNormal)
                )[0]
            )

            algorithm = "non-atomic" if self.use_non_atomic_loss else "atomic"
            print(f"Using SNPE-C with {algorithm} loss")

            if self.use_non_atomic_loss:
                # Take care of z-scoring, pre-compute and store prior terms.
                self._set_state_for_mog_proposal()

        return super().train_base_p3(**kwargs) # type: ignore
    


class MyPosteriorEstimator_NPE(PosteriorEstimator):
    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Union[str, Callable] = "maf",
        device: str = "gpu",
        logging_level: Union[int, str] = "INFO",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,    
    ):
        kwargs = del_entries(locals(), entries=("self", "__class__"))
        super().__init__(**kwargs)
    
    @staticmethod
    def _maybe_show_progress(show, epoch, starting_time, train_log_prob, val_log_prob):
        if show:
            print("\r", f"Epochs trained: {epoch:5}. Time elapsed {(time.time()-starting_time)/ 60:6.2f}min ||  log_prob train: {train_log_prob:.2f} val: {val_log_prob:.2f}", end="")
    
    def _converged(self, epoch: int, stop_after_epochs: int) -> bool:
        """Return whether the training converged yet and save best model state so far.

        Checks for improvement in validation performance over previous epochs.

        Args:
            epoch: Current epoch in training.
            stop_after_epochs: How many fruitless epochs to let pass before stopping.

        Returns:
            Whether the training has stopped improving, i.e. has converged.
        """
        converged = False

        assert self._neural_net is not None
        neural_net = self._neural_net

        # (Re)-start the epoch count with the first epoch or any improvement.
        if epoch == 0 or self._val_log_prob > self._best_val_log_prob:
            self._best_val_log_prob = self._val_log_prob
            self._epochs_since_last_improvement = 0
            self._best_model_state_dict = deepcopy(neural_net.state_dict())
            self._best_model_from_epoch = epoch
        else:
            self._epochs_since_last_improvement += 1

        # If no validation improvement over many epochs, stop training.
        if self._epochs_since_last_improvement > stop_after_epochs - 1:
            # neural_net.load_state_dict(self._best_model_state_dict)
            converged = True
            self._neural_net.load_state_dict(self._best_model_state_dict)
            self._val_log_prob = self._best_val_log_prob
            
            self._epochs_since_last_improvement = 0
            
        return converged
    
    
    def append_simulations_for_run(
        self,
        theta,
        x,
        current_round: int = 0,
        exclude_invalid_x: Optional[bool] = None,
        data_device: Optional[str] = None,
    ):
        """ update theta and x for the current round
        """
        if exclude_invalid_x is None:
            if current_round == 0:
                exclude_invalid_x = True
            else:
                exclude_invalid_x = False

        if data_device is None:
            data_device = self._device

        theta, x = validate_theta_and_x(
            theta, x, data_device=data_device, training_device=self._device
        )

        is_valid_x, num_nans, num_infs = handle_invalid_x(
            x, exclude_invalid_x=exclude_invalid_x
        )

        x = x[is_valid_x]
        theta = theta[is_valid_x]

        # Check for problematic z-scoring
        warn_if_zscoring_changes_data(x)
        if (
            type(self).__name__ == "SNPE_C"
            and current_round > 0
            and not self.use_non_atomic_loss
        ):
            nle_nre_apt_msg_on_invalid_x(
                num_nans, num_infs, exclude_invalid_x, "Multiround SNPE-C (atomic)"
            )
        else:
            npe_msg_on_invalid_x(
                num_nans, num_infs, exclude_invalid_x, "Single-round NPE"
            )

        prior_masks = mask_sims_from_prior(int(current_round > 0), theta.size(0))

        old_theta   = self._theta_roundwise[current_round]
        old_x       = self._x_roundwise[current_round]
        old_masks   = self._prior_masks[current_round]
        
        self._theta_roundwise[current_round] = torch.cat((old_theta, theta), dim=0)
        self._x_roundwise[current_round]     = torch.cat((old_x, x), dim=0)
        self._prior_masks[current_round]     = torch.cat((old_masks, prior_masks), dim=0)
        
        return self
    
    def get_dataloaders(
        self,
        starting_round: int = 0,
        training_batch_size: int = 50,
        validation_fraction: float = 0.1,
        resume_training: bool = False,
        seed: Optional[int] = 100,
        dataloader_kwargs: Optional[dict] = None,
        loading_mode='random', # 'fixed_permutation' or 'random_permutation'
    ) -> Tuple[data.DataLoader, data.DataLoader]:
        """Return dataloaders for training and validation.

        Args:
            dataset: holding all theta and x, optionally masks.
            training_batch_size: training arg of inference methods.
            resume_training: Whether the current call is resuming training so that no
                new training and validation indices into the dataset have to be created.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn).

        Returns:
            Tuple of dataloaders for training and validation.

        """
        print(f'\n--- get_dataloaders ---\nloading_mode: {loading_mode}')
        dataset = Choice_Sampled_2D_Dataset(
            config = self.config, 
            chosen_set_names = ['set_0', 'set_1'], 
            num_chosen_theta_each_set = self.config.dataset.train_num_theta, 
            chosen_dur=self.config.experiment_settings.chosen_dur_list,
            theta_chosen_mode='random',
            permutation_mode='random',
        )
        
        # load and show one example of the dataset
        i = 1
        #  print dataset size
        print(f'\n--- dataset ---\nsize [{len(dataset)}]',
              f'\none examples of the dataset [{i}]:',
              f'\n| theta[{i}] info:', f'shape {dataset[i][1].shape}, dtype: {dataset[i][1].dtype}, device: {dataset[i][1].device}',
              f'\n| x[{i}] info:', f'shape {dataset[i][0].shape}, dtype: {dataset[i][0].dtype}, device: {dataset[i][0].device}',
             )
        # Get total number of training examples.
        num_examples = len(dataset)
        # Select random train and validation splits from (theta, x) pairs.
        num_training_examples = int((1 - validation_fraction) * num_examples)
        num_validation_examples = num_examples - num_training_examples

        # if not resume_training:
        # Seperate indicies for training and validation
        permuted_indices = torch.randperm(num_examples)
        self.train_indices, self.val_indices = (
            permuted_indices[:num_training_examples],
            permuted_indices[num_training_examples:],
        )

        # Create training and validation loaders using a subset sampler.
        # Intentionally use dicts to define the default dataloader args
        # Then, use dataloader_kwargs to override (or add to) any of these defaults
        # https://stackoverflow.com/questions/44784577/in-method-call-args-how-to-override-keyword-argument-of-unpacked-dict
        train_loader_kwargs = {
            "batch_size": min(training_batch_size, num_training_examples),
            "drop_last": True,
            "sampler": SubsetRandomSampler(self.train_indices.tolist()),
        }
        val_loader_kwargs = {
            "batch_size": min(training_batch_size, num_validation_examples),
            "shuffle": False,
            "drop_last": True,
            "sampler": SubsetRandomSampler(self.val_indices.tolist()),
        }
        if dataloader_kwargs is not None:
            train_loader_kwargs = dict(train_loader_kwargs, **dataloader_kwargs)
            val_loader_kwargs = dict(val_loader_kwargs, **dataloader_kwargs)

        print(f'\n--- data loader ---\ntrain_loader_kwargs: {train_loader_kwargs}')
        print(f'val_loader_kwargs: {val_loader_kwargs}')
        
        g = torch.Generator()
        g.manual_seed(seed)
        
        train_loader = data.DataLoader(dataset, generator=g, **train_loader_kwargs)
        val_loader = data.DataLoader(dataset, generator=g, **val_loader_kwargs)

        # + load and show some examples of the dataloader
        # train_batch = next(iter(train_loader))
        # val_batch = next(iter(val_loader))
        # print('train batch: ', train_batch)
        # print('val batch: '  , val_batch ) 
        
        return train_loader, val_loader
    
    def train_base(
        self,
        config, 
        num_atoms: int = 10,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        
        calibration_kernel: Optional[Callable] = None,
        resume_training: bool = False,
        force_first_round_loss: bool = False,
        discard_prior_samples: bool = False,
        retrain_from_scratch: bool = False,
        show_train_summary: bool = True,
        
        seed: Optional[int] = 100,
        dataloader_kwargs: Optional[dict] = None,
        debug: Optional[bool] = False,
    ) -> nn.Module:
        r"""Return density estimator that approximates the distribution $p(\theta|x)$.

        Args:
            training_batch_size: Training batch size.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            stop_after_epochs: The number of epochs to wait for improvement on the
                validation set before terminating training.
            max_num_epochs: Maximum number of epochs to run. If reached, we stop
                training even when the validation loss is still decreasing. Otherwise,
                we train until validation loss increases (see also `stop_after_epochs`).
            clip_max_norm: Value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.
            calibration_kernel: A function to calibrate the loss with respect to the
                simulations `x`. See Lueckmann, Gonçalves et al., NeurIPS 2017.
            #TODO resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If `True`, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time `.train()` was called.
            #TODO force_first_round_loss: If `True`, train with maximum likelihood,
                i.e., potentially ignoring the correction for using a proposal
                distribution different from the prior.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less targeted
                samples.
            retrain_from_scratch: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.
            show_train_summary: Whether to print the number of epochs and validation
                loss after the training.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn)

        Returns:
            Density estimator that approximates the distribution $p(\theta|x)$.
        """
        self._summary["learning_rates"] = []
        # Load data from most recent round.
        self._round = max(self._data_round_index)

        if self._round == 0 and self._neural_net is not None:
            assert force_first_round_loss, (
                "You have already trained this neural network. After you had trained "
                "the network, you again appended simulations with `append_simulations"
                "(theta, x)`, but you did not provide a proposal. If the new "
                "simulations are sampled from the prior, you can set "
                "`.train(..., force_first_round_loss=True`). However, if the new "
                "simulations were not sampled from the prior, you should pass the "
                "proposal, i.e. `append_simulations(theta, x, proposal)`. If "
                "your samples are not sampled from the prior and you do not pass a "
                "proposal and you set `force_first_round_loss=True`, the result of "
                "SNPE will not be the true posterior. Instead, it will be the proposal "
                "posterior, which (usually) is more narrow than the true posterior."
            )

        # Calibration kernels proposed in Lueckmann, Gonçalves et al., 2017.
        if calibration_kernel is None:
            calibration_kernel = lambda x: ones([len(x)], device=self._device)

        # Starting index for the training set (1 = discard round-0 samples).
        start_idx = int(discard_prior_samples and self._round > 0)

        # For non-atomic loss, we can not reuse samples from previous rounds as of now.
        # SNPE-A can, by construction of the algorithm, only use samples from the last
        # round. SNPE-A is the only algorithm that has an attribute `_ran_final_round`,
        # so this is how we check for whether or not we are using SNPE-A.
        if self.use_non_atomic_loss or hasattr(self, "_ran_final_round"):
            start_idx = self._round

        # Set the proposal to the last proposal that was passed by the user. For
        # atomic SNPE, it does not matter what the proposal is. For non-atomic
        # SNPE, we only use the latest data that was passed, i.e. the one from the
        # last proposal.
        proposal = self._proposal_roundwise[-1]

        train_loader, val_loader = self.get_dataloaders(
            start_idx,
            training_batch_size,
            validation_fraction,
            resume_training,
            seed=seed,
            dataloader_kwargs=dataloader_kwargs,
        )
        # First round or if retraining from scratch:
        # Call the `self._build_neural_net` with the rounds' thetas and xs as
        # arguments, which will build the neural network.
        # This is passed into NeuralPosterior, to create a neural posterior which
        # can `sample()` and `log_prob()`. The network is accessible via `.net`.
        if self._neural_net is None or retrain_from_scratch:

            # Get theta,x to initialize NN
            # theta, x, _ = self.get_simulations(starting_round=start_idx)
            x, theta = next(iter(train_loader))
            # Use only training data for building the neural net (z-scoring transforms)
            self._neural_net = self._build_neural_net(
                theta[0:3].to("cpu"),
                x[0:3].to("cpu"),
            )
            # self._x_shape = x_shape_from_simulation(x.to("cpu"))
            
            print('\nfinished build network')
            print(f'\n{self._neural_net}')
            test_posterior_net_for_multi_d_x(
                self._neural_net,
                theta.to("cpu"),
                x.to("cpu"),
            )

            del theta, x

        # Move entire net to device for training.
        self._neural_net.to(self._device)

        if not resume_training:
            self.optimizer = optim.Adam(
                list(self._neural_net.parameters()), lr=learning_rate
            )
            self.epoch, self._val_log_prob = 0, float("-Inf")

        epoch_start_time = time.time()
        while self.epoch <= max_num_epochs and not self._converged(
            self.epoch, stop_after_epochs
        ):
            starting_time = time.time()
            # Train for a single epoch.
            self._neural_net.train()
            train_log_probs_sum = 0
            for batch in train_loader:
                self.optimizer.zero_grad()
                # Get batches on current device.
                x_batch, theta_batch = (
                    batch[0].to(self._device),
                    batch[1].to(self._device),
                    # batch[2].to(self._device),
                )
                masks_batch = torch.ones_like(theta_batch[:, 0]).to(self._device)

                train_losses = self._loss(
                    theta_batch,
                    x_batch,
                    masks_batch,
                    proposal,
                    calibration_kernel,
                    force_first_round_loss=force_first_round_loss,
                )
                train_loss = torch.mean(train_losses)
                train_log_probs_sum -= train_losses.sum().item()

                train_loss.backward()
                if clip_max_norm is not None:
                    clip_grad_norm_(
                        self._neural_net.parameters(), max_norm=clip_max_norm
                    )
                self.optimizer.step()

            self.epoch += 1

            train_log_prob_average = train_log_probs_sum / (
                len(train_loader) * train_loader.batch_size  # type: ignore
            )
            current_learning_rate  = self.optimizer.param_groups[0]['lr']
            
            self._summary["training_log_probs"].append(train_log_prob_average)
            self._summary["learning_rates"].append(current_learning_rate)
            # Calculate validation performance.
            self._neural_net.eval()
            val_log_prob_sum = 0

            with torch.no_grad():
                for batch in val_loader:
                    x_batch, theta_batch = (
                        batch[0].to(self._device),
                        batch[1].to(self._device),
                        # batch[2].to(self._device),
                    )
                    masks_batch = torch.ones_like(theta_batch[:, 0]).to(self._device)
                    # Take negative loss here to get validation log_prob.
                    val_losses = self._loss(
                        theta_batch,
                        x_batch,
                        masks_batch,
                        proposal,
                        calibration_kernel,
                        force_first_round_loss=force_first_round_loss,
                    )
                    val_log_prob_sum -= val_losses.sum().item()

            # Take mean over all validation samples.
            self._val_log_prob = val_log_prob_sum / (
                len(val_loader) * val_loader.batch_size  # type: ignore
            )
            # Log validation log prob for every epoch.
            self._summary["validation_log_probs"].append(self._val_log_prob)
            self._summary["epoch_durations_sec"].append(time.time() - epoch_start_time)

            self._maybe_show_progress(self._show_progress_bars, self.epoch, starting_time, train_log_prob_average, self._val_log_prob)
            self._plot_training_curve()
            
            if debug: 
                break
            
        # self._report_convergence_at_end(self.epoch, stop_after_epochs, max_num_epochs)
        # self._val_log_prob = self._best_val_log_prob
        
        # Update summary.
        self._summary["epochs_trained"].append(self.epoch)
        self._summary["best_validation_log_prob"].append(self._best_val_log_prob)

        # Update tensorboard and summary dict.
        self._summarize(round_=self._round)

        # Update description for progress bar.
        if show_train_summary:
            print(self._describe_round(self._round, self._summary))

        # load best model from state dict
        self._neural_net.load_state_dict(self._best_model_state_dict)
        self._val_log_prob = self._best_val_log_prob
        self._epochs_since_last_improvement = 0
        
        # Avoid keeping the gradients in the resulting network, which can
        # cause memory leakage when benchmarking.
        self._neural_net.zero_grad(set_to_none=True)

        return self, deepcopy(self._neural_net)
    
    def _describe_round(self, round_: int, summary: Dict[str, list]) -> str:
        epochs = summary["epochs_trained"][-1]
        best_validation_log_prob = summary["best_validation_log_prob"][-1]

        description = f"""
        -------------------------
        ||||| ROUND {round_} STATS |||||:
        -------------------------
        Epochs trained: {epochs}
        Best validation performance: {best_validation_log_prob:.4f}, from epoch {self._best_model_from_epoch:5}
        Model from best epoch {self._best_model_from_epoch} is loaded for further training
        -------------------------
        """

        return description

    def _plot_training_curve(self):
        log_dir         = self.config.log_dir
        duration        = np.array(self._summary["epoch_durations_sec"])
        train_log_probs = self._summary["training_log_probs"]
        val_log_probs   = self._summary["validation_log_probs"]
        learning_rates  = self._summary["learning_rates"]
        best_val_log_prob = self._best_val_log_prob
        best_val_log_prob_epoch = self._best_model_from_epoch
        
        plt.tight_layout()
        
        fig, axes = plt.subplots(2,1, figsize=(16,10))
        fig.subplots_adjust(hspace=0.3)
        
        # plot learning rate
        ax0 = axes[0]
        ax0.plot(learning_rates, '-', label='lr', lw=2)
        ax0.plot(best_val_log_prob_epoch, learning_rates[best_val_log_prob_epoch-1], 'v', color='tab:red', lw=2) # type: ignore

        ax0.set_xlabel('epochs')
        ax0.set_ylabel('learning rate')
        ax0.grid(alpha=0.2)
        ax0.set_title('training curve')

        ax1 = axes[1]
        ax1.plot(train_log_probs, '.-', label='training', alpha=0.8, lw=2, color='tab:blue', ms=0.1)
        ax1.plot(val_log_probs, '.-', label='validation', alpha=0.8, lw=2, color='tab:orange', ms=0.1)
        if "test_log_probs" in self._summary.keys():
            test_log_probs = self._summary["test_log_probs"]
            ax1.plot(test_log_probs, '.-', label='test', alpha=0.8, lw=2, color='tab:brown', ms=0.1)
        
        # find best val log prob, and plot it
        best_val_log_prob = max(val_log_probs)
        best_val_log_prob_epoch = np.argmax(val_log_probs)
        
        ax1.plot(best_val_log_prob_epoch, best_val_log_prob, 'v', color='red', lw=2)
        ax1.text(best_val_log_prob_epoch, best_val_log_prob+0.02, f'{best_val_log_prob:.2f}', color='red', fontsize=10, ha='center', va='bottom') # type: ignore
        # ax1.set_ylim(log_probs_lower_bound, max(val_log_probs)+0.2)
        
        ax1.legend()
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('log_prob')
        ax1.grid(alpha=0.2)

        ax2 = ax1.twiny()
        ax2.plot((duration-duration[0])/60/60, max(val_log_probs)*np.ones_like(val_log_probs), '-', alpha=0)
        ax2.set_xlabel('time (hours)')

        # save the figure
        plt.savefig(f'{log_dir}/training_curve.png')
        # print('saved training curve')
        plt.close()
    
class MySNPE_C_NPE(SNPE_C, MyPosteriorEstimator_NPE):
    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Union[str, Callable] = "maf",
        device: str = "gpu",
        logging_level: Union[int, str] = "INFO",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,    
    ):
        kwargs = del_entries(locals(), entries=("self", "__class__"))
        # print(kwargs)
        super().__init__(**kwargs)
    
    def train(
        self,
        config,
        num_atoms: int = 10,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        calibration_kernel: Optional[Callable] = None,
        resume_training: bool = False,
        force_first_round_loss: bool = False,
        discard_prior_samples: bool = False,
        use_combined_loss: bool = False,
        retrain_from_scratch: bool = False,
        show_train_summary: bool = False,
        seed: Optional[int] = 100,
        dataloader_kwargs: Optional[Dict] = None,
    ) -> nn.Module:
        r"""Return density estimator that approximates the distribution $p(\theta|x)$.

        Args:
            num_atoms: Number of atoms to use for classification.
            training_batch_size: Training batch size.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            stop_after_epochs: The number of epochs to wait for improvement on the
                validation set before terminating training.
            max_num_epochs: Maximum number of epochs to run. If reached, we stop
                training even when the validation loss is still decreasing. Otherwise,
                we train until validation loss increases (see also `stop_after_epochs`).
            clip_max_norm: Value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.
            calibration_kernel: A function to calibrate the loss with respect to the
                simulations `x`. See Lueckmann, Gonçalves et al., NeurIPS 2017.
            resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If `True`, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time `.train()` was called.
            force_first_round_loss: If `True`, train with maximum likelihood,
                i.e., potentially ignoring the correction for using a proposal
                distribution different from the prior.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less targeted
                samples.
            use_combined_loss: Whether to train the neural net also on prior samples
                using maximum likelihood in addition to training it on all samples using
                atomic loss. The extra MLE loss helps prevent density leaking with
                bounded priors.
            retrain_from_scratch: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.
            show_train_summary: Whether to print the number of epochs and validation
                loss and leakage after the training.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn)

        Returns:
            Density estimator that approximates the distribution $p(\theta|x)$.
        """

        # WARNING: sneaky trick ahead. We proxy the parent's `train` here,
        # requiring the signature to have `num_atoms`, save it for use below, and
        # continue. It's sneaky because we are using the object (self) as a namespace
        # to pass arguments between functions, and that's implicit state management.
        self.config     = config
        self._num_atoms = num_atoms
        self._use_combined_loss = use_combined_loss
        kwargs = del_entries(
            locals(), entries=("self", "__class__", "num_atoms", "use_combined_loss")
        )

        self._round = max(self._data_round_index)

        if self._round > 0:
            # Set the proposal to the last proposal that was passed by the user. For
            # atomic SNPE, it does not matter what the proposal is. For non-atomic
            # SNPE, we only use the latest data that was passed, i.e. the one from the
            # last proposal.
            proposal = self._proposal_roundwise[-1]
            self.use_non_atomic_loss = (
                isinstance(proposal, DirectPosterior)
                and isinstance(proposal.posterior_estimator._distribution, mdn)
                and isinstance(self._neural_net._distribution, mdn)
                and check_dist_class(
                    self._prior, class_to_check=(Uniform, MultivariateNormal)
                )[0]
            )

            algorithm = "non-atomic" if self.use_non_atomic_loss else "atomic"
            print(f"Using SNPE-C with {algorithm} loss")

            if self.use_non_atomic_loss:
                # Take care of z-scoring, pre-compute and store prior terms.
                self._set_state_for_mog_proposal()

        return super().train_base(**kwargs)
    

