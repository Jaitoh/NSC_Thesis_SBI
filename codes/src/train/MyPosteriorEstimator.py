import os
import time
import h5py
from copy import deepcopy
import numpy as np

import torch
from torch import nn
from torch import Tensor, nn, ones, optim
from torch.distributions import Distribution, MultivariateNormal, Uniform
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, CosineAnnealingLR
from pyknos.mdn.mdn import MultivariateGaussianMDN as mdn
from typing import Any, Callable, Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
from tqdm import tqdm
# from torchinfo import summary

from sbi.inference import SNPE_C
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.snpe.snpe_base import PosteriorEstimator
from sbi.utils import (
    check_dist_class,
    del_entries,
    test_posterior_net_for_multi_d_x,
    x_shape_from_simulation,
    del_entries,
    validate_theta_and_x,
    handle_invalid_x,
    warn_if_zscoring_changes_data,
    nle_nre_apt_msg_on_invalid_x,
    npe_msg_on_invalid_x,
)   
from sbi.utils.sbiutils import mask_sims_from_prior

import signal
import sys
sys.path.append('./src')

from train.MyData import My_Dataset_Mem, Data_Prefetcher, My_Chosen_Sets
from utils.train import (
    plot_posterior_with_label,
)
from utils.resource import(
    print_mem_info
)


do_print_mem = 1
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:2048'

def signal_handler(sig, frame):
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

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
        
        log_dir,
        config, 
        
        seed,
        prior_limits,
        
        dataset_kwargs,
        dataloader_kwargs,
        training_kwargs,
        
        do_print_memory_usage=do_print_mem,
        continue_from_checkpoint=None,
        
    ):
        """ train base model (first round for SNPE)
        
        """
        self.log_dir = log_dir
        self.config  = config
        self.prior_limits = prior_limits
        
        chosen_dur_trained_in_sequence  = dataset_kwargs['chosen_dur_trained_in_sequence']
        clip_max_norm                   = training_kwargs['clip_max_norm']
        print_freq                      = training_kwargs['print_freq']
        use_data_prefetcher             = dataset_kwargs['use_data_prefetcher']
        self.use_data_prefetcher        = use_data_prefetcher
        self.batch_counter = 0
        self.epoch_counter = 0
        self.dset_counter  = 0
        
        self.train_data_set_name_list = []
        self.val_data_set_name_list   = []
        
        val_set_names = self._get_val_set_names(dataset_kwargs)
        
        try:
            for self.run, chosen_dur in enumerate(chosen_dur_trained_in_sequence):
                
                # init log prob
                self.dset  = 0
                self.epoch = 0
                self._epoch_of_last_dset = 0
                self._val_log_prob, self._val_log_prob_dset = float("-Inf"), float("-Inf")
                self._best_val_log_prob, self._best_val_log_prob_dset = float("-Inf"), float("-Inf")
                
                # prepare dataset and dataloader
                # train_loader, val_loader = self._prepare_data_loader(dataset_kwargs, dataloader_kwargs, seed, chosen_dur)
                val_loader = self._get_val_loader(val_set_names, dataset_kwargs, dataloader_kwargs, seed, chosen_dur)
                train_loader = self._get_train_loader(val_set_names, dataset_kwargs, dataloader_kwargs, seed, chosen_dur)
                
                if use_data_prefetcher:
                    train_prefetcher, val_prefetcher = self._get_data_prefetcher(train_loader, val_loader)
                    print('\ntraining batch ...', end=' ')
                    x, theta = self._load_one_data(train_prefetcher, use_data_prefetcher)
                    print('\nvalidation batch ...', end=' ')
                    x_val, theta_val = self._load_one_data(val_prefetcher, use_data_prefetcher)
                else:
                    print('\ntraining batch ...', end=' ')
                    x, theta = self._load_one_data(train_loader, use_data_prefetcher)
                    print('\nvalidation batch ...', end=' ')
                    x_val, theta_val = self._load_one_data(val_loader, use_data_prefetcher)
                
                self._collect_posterior_sets(x, theta, x_val, theta_val)
                
                # init neural net, move to device only once
                if self.run == 0 and self.dset_counter == 0:
                    self._init_neural_net(x, theta, continue_from_checkpoint=continue_from_checkpoint)
                
                
                while self.dset <= training_kwargs['max_num_dsets'] and not self._converged_dset(training_kwargs['stop_after_dsets'], training_kwargs['improvement_threshold'], training_kwargs['min_num_dsets']):
                    
                    # init optimizer and scheduler
                    self._init_optimizer(training_kwargs)
                    
                    print(f'\n\n=== run {self.run}, chosen_dur {chosen_dur}, dset {self.dset} ===')
                    print_mem_info(f"\n{'gpu memory usage after loading dataset':46}", do_print_memory_usage)
                    
                    while self.epoch <= training_kwargs['max_num_epochs'] and not self._converged(self.epoch, training_kwargs['stop_after_epochs'], training_kwargs['improvement_threshold'], training_kwargs['min_num_epochs']):
                        
                        # train and log one epoch
                        self._neural_net.train()
                        if use_data_prefetcher:
                            epoch_start_time, train_log_probs_sum = self._train_one_epoch(clip_max_norm, print_freq, train_prefetcher, x, theta)
                        else: 
                            epoch_start_time, train_log_probs_sum = self._train_one_epoch(clip_max_norm, print_freq, train_loader, x, theta)
                            
                        train_log_prob_average = self._train_one_epoch_log(train_log_probs_sum)
                        
                        # validate and log 
                        self._neural_net.eval()
                        with torch.no_grad():
                            val = val_prefetcher if use_data_prefetcher else val_loader
                            self._val_log_prob = self._val_one_epoch(val)
                            self._val_one_epoch_log(epoch_start_time)
                            print_mem_info(f"{'gpu memory usage after validation':46}", do_print_memory_usage)
                                
                            # check posterior behavior after each epoch
                            self._show_epoch_progress(self.epoch, epoch_start_time, train_log_prob_average, self._val_log_prob)
                            # self._posterior_behavior_log(config, prior_limits, log_dir)
                            print_mem_info(f"{'gpu memory usage after posterior behavior log':46}", do_print_memory_usage)
                            
                            # fetcher same dataset for next epoch
                            if self.use_data_prefetcher:
                                train_prefetcher, val_prefetcher = self._get_data_prefetcher(train_loader, val_loader)
                            print_mem_info(f"\n\n{'gpu memory usage after prefetcher':46}", do_print_memory_usage)
                            print()
                            
                            self.epoch += 1
                            self.epoch_counter += 1
                        
                        if training_kwargs['scheduler'] == 'ReduceLROnPlateau':
                            self.scheduler.step(self._val_log_prob)
                        else:
                            self.scheduler.step()
                            
                    if self.use_data_prefetcher:
                        del train_prefetcher, val_prefetcher, train_loader, self.train_dataset
                    else:
                        del train_loader, self.train_dataset
                        
                    torch.cuda.empty_cache()
                    print(self._describe_dset())
                    
                    # log info for this dset
                    self._summary_writer.add_scalar(f"run{self.run}/best_val_epoch_of_dset", self._best_model_from_epoch, self.dset_counter)
                    self._summary_writer.add_scalar(f"run{self.run}/best_val_log_prob_of_dset", self._best_val_log_prob, self.dset_counter)
                    self._summary_writer.add_scalar(f"run{self.run}/current_dset", self.dset_counter, self.dset_counter)
                    self._summary_writer.add_scalar(f"run{self.run}/num_chosen_dset", self.num_chosen_sets, self.dset_counter)
                    

                    # update dset info
                    self._val_log_prob_dset = self._best_val_log_prob
                    
                    # load data for next dset
                    self.dset         += 1
                    self.dset_counter += 1
                    
                    # load new dataset for the next dset
                    with torch.no_grad():
                        # train_loader, val_loader = self._prepare_data_loader(dataset_kwargs, dataloader_kwargs, seed, chosen_dur)
                        # val_loader = self._get_val_loader(val_set_names, dataset_kwargs, dataloader_kwargs, seed, chosen_dur)
                        train_loader = self._get_train_loader(val_set_names, dataset_kwargs, dataloader_kwargs, seed, chosen_dur)
                        if use_data_prefetcher:
                            train_prefetcher, val_prefetcher = self._get_data_prefetcher(train_loader, val_loader)
                            print('\nloading one training batch...')
                            x, theta = self._load_one_data(train_prefetcher, use_data_prefetcher)
                        else:
                            print('\nloading one training batch...')
                            x, theta = self._load_one_data(train_loader, use_data_prefetcher)
                
                if use_data_prefetcher:
                    del train_prefetcher, val_prefetcher, train_loader, x, theta
                else:
                    del train_loader, x, theta
                torch.cuda.empty_cache()
            
            # Avoid keeping the gradients in the resulting network, which can
            # cause memory leakage when benchmarking. save the network
            self._neural_net.zero_grad(set_to_none=True)
            torch.save(self._neural_net, os.path.join(log_dir, f"model/round_{self._round}_model.pt"))
            
            # log training curve to tensorboard
            figure = plt.figure(figsize=(16, 10))
            plt.plot(self._summary["training_log_probs"], label="training")
            plt.plot(self._summary["validation_log_probs"], label="validation")
            plt.xlabel("Epoch")
            plt.ylabel("Log prob")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, f"model/round_{self._round}_training_curve.png"))
            self._summary_writer.add_figure("training_curve", figure, self._round)
            plt.close(figure)
            
            return self, deepcopy(self._neural_net)

        # except Exception as e:
        #         print(f"An error occurred: {e}")
        finally:
            # Release GPU resources
            train_loader        = None
            train_prefetcher    = None
            val_loader          = None
            val_prefetcher      = None
            
            torch.cuda.empty_cache()
            print('cuda cache emptied')
            
            # clear self
            self._do_clear()
            print('self cleared')
        
    def _do_clear(self):
        # clear all attributes and variables
        self._neural_net        = None
        self._optimizer         = None
        self._scheduler         = None
        self._train_log_prob    = None
        self._val_log_prob      = None
        self._summary           = None
        self._summary_writer    = None
        self._round             = None
        self._best_model        = None
        self._best_model_from_epoch = None
        self._best_val_log_prob = None
        self._val_log_prob_dset = None
        self._best_val_log_prob_dset = None
        self._best_model_dset   = None
        self._best_model_from_epoch_dset = None
        self._best_val_log_prob_dset = None
        self._best_val_log_prob_from_epoch_dset = None
        self._best_val_log_prob_from_epoch = None
        
        
    def _get_val_set_names(self, dataset_kwargs):
        
        all_set_names = self._get_all_set_names(dataset_kwargs)
        
        # check if validation set is float number 
        if isinstance(dataset_kwargs['validation_fraction'], float):
            num_val_sets   = int(dataset_kwargs['validation_fraction']*len(all_set_names))
            val_set_names  = np.random.choice(all_set_names, num_val_sets, replace=False)
        if isinstance(dataset_kwargs['validation_fraction'], list):
            val_set_names  = [f'set_{i}' for i in dataset_kwargs['validation_fraction']]
        # train_set_names   = list(set(all_set_names) - set(val_set_names))
        
        return val_set_names

    def _get_all_set_names(self, dataset_kwargs):
        
        # get all available set names
        data_path = dataset_kwargs['data_path']
        f = h5py.File(data_path, 'r', libver='latest', swmr=True)
        all_set_names = list(f.keys())
        num_total_sets = len(all_set_names)
        f.close()
        
        # select a subset of sets
        num_max_sets = dataset_kwargs['num_max_sets']
        num_max_sets = min(num_max_sets, num_total_sets)
        all_set_names = all_set_names[:num_max_sets]
        print(f'\n === chosen {len(all_set_names)} sets from {num_total_sets} sets === \n')
        
        return all_set_names
        
    def _get_val_loader(self, val_set_names, dataset_kwargs, dataloader_kwargs, seed, chosen_dur):
        """ the val loader keeps the same for each dset, and each run
        """
        self.val_dataset = My_Chosen_Sets(
            data_path   = dataset_kwargs['data_path'],
            config      = dataset_kwargs['config'],
            chosen_set_names = val_set_names,
            chosen_dur  = chosen_dur,
            crop_dur    = dataset_kwargs['crop_dur'],
        )
        
        batch_size = dataloader_kwargs['batch_size']
        num_validation_examples = len(self.val_dataset)
        val_indices = torch.randperm(num_validation_examples)
        
        val_loader_kwargs = {
            "batch_size": min(batch_size, num_validation_examples),
            "shuffle"   : False,
            "drop_last" : True,
            "sampler"   : SubsetRandomSampler(val_indices.tolist()),
            "pin_memory": True,
        }
        if dataloader_kwargs is not None:
            val_loader_kwargs = dict(val_loader_kwargs, **dataloader_kwargs)
        # print(f'\n--- data loader ---')
        print(f'--> val_loader_kwargs: {val_loader_kwargs}')
        
        g = torch.Generator()
        g.manual_seed(seed) # no dset here
        
        val_loader   = data.DataLoader(self.val_dataset, generator=g, **val_loader_kwargs)
        
        return val_loader
        
    
    def _get_train_loader(self, val_set_names, dataset_kwargs, dataloader_kwargs, seed, chosen_dur):
        """ the train loader updates each dset
        """
        # get current all dataset names
        all_set_names = self._get_all_set_names(dataset_kwargs)
        all_train_set_names = list(set(all_set_names) - set(val_set_names))
        
        # first dset is trained with first element of the num_chosen_set list, 2nd -> 2nd ...
        num_chosen_sets = dataset_kwargs['num_chosen_sets'][self.dset%len(dataset_kwargs['num_chosen_sets'])]
        if self.dset>len(dataset_kwargs['num_chosen_sets'])-1:
            num_chosen_sets = dataset_kwargs['num_chosen_sets'][-1]
        self.num_chosen_sets = num_chosen_sets
        print(f'train_loader - num_chosen_sets: {num_chosen_sets}')
        assert num_chosen_sets <= len(all_train_set_names), 'not enough training sets'
        chosen_train_set_names = np.random.choice(all_train_set_names, num_chosen_sets, replace=False)
        
        self.train_dataset = My_Chosen_Sets( #TODO del self.train_dataset
            data_path   = dataset_kwargs['data_path'],
            config      = dataset_kwargs['config'],
            chosen_set_names = chosen_train_set_names,
            chosen_dur  = chosen_dur,
            crop_dur    = dataset_kwargs['crop_dur'],
        )
        
        batch_size = dataloader_kwargs['batch_size']
        num_train_examples = len(self.train_dataset)
        train_indices = torch.randperm(num_train_examples)
        
        train_loader_kwargs = {
            "batch_size": min(batch_size, num_train_examples),
            "drop_last" : True,
            "sampler"   : SubsetRandomSampler(train_indices.tolist()),
            "pin_memory": True,
        }
        if dataloader_kwargs is not None:
            train_loader_kwargs = dict(train_loader_kwargs, **dataloader_kwargs)

        print(f'--> train_loader_kwargs: {train_loader_kwargs}')
        g = torch.Generator()
        g.manual_seed(seed+self.dset)
        
        train_loader = data.DataLoader(self.train_dataset, generator=g, **train_loader_kwargs)
        
        self.num_train_batches = len(train_loader)
        print(f'num_chosen_sets of train_loader: {num_chosen_sets}')
        print(f'number of batches in the training dataset: {self.num_train_batches}')
        
        return train_loader
    
    def _prepare_data_loader(self, dataset_kwargs, dataloader_kwargs, seed, chosen_dur):
        
        self.dataset = My_Dataset_Mem(
            data_path           = dataset_kwargs['data_path'],
            config              = dataset_kwargs['config'],
            chosen_dur          = chosen_dur, 
        )
        
        train_loader, val_loader = self._get_dataloaders(
            dataset = self.dataset,
            seed    = seed,
            dataloader_kwargs=dataloader_kwargs,
            dataset_kwargs=dataset_kwargs,
        )
        
        self.num_train_batches = len(train_loader)
        self.num_val_batches   = len(val_loader)
        
        print(f'\nnumber of batches in the training dataset: {self.num_train_batches}')
        print(f'number of batches in the validation dataset: {self.num_val_batches}')
        
        return train_loader, val_loader
    
    def _get_dataloaders(
        self,
        dataset,
        seed: int,
        dataloader_kwargs: Optional[dict] = None,
        dataset_kwargs: Optional[dict] = None,
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
        training_batch_size = dataloader_kwargs['batch_size']
        validation_fraction = dataset_kwargs['validation_fraction']
        
        # Get total number of training examples.
        num_examples = len(dataset)
        # Select random train and validation splits from (theta, x) pairs.
        num_training_examples = int((1 - validation_fraction) * num_examples)
        num_validation_examples = num_examples - num_training_examples

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
            "drop_last" : True,
            "sampler"   : SubsetRandomSampler(self.train_indices.tolist()),
            "pin_memory": True,
        }
        val_loader_kwargs = {
            "batch_size": min(training_batch_size, num_validation_examples),
            "shuffle"   : False,
            "drop_last" : True,
            "sampler"   : SubsetRandomSampler(self.val_indices.tolist()),
            "pin_memory": True,
        }
        if dataloader_kwargs is not None:
            train_loader_kwargs = dict(train_loader_kwargs, **dataloader_kwargs)
            val_loader_kwargs = dict(val_loader_kwargs, **dataloader_kwargs)

        print(f'\n--- data loader ---\nfinal train_loader_kwargs: \n{train_loader_kwargs}')
        print(f'final val_loader_kwargs: \n{val_loader_kwargs}')
        
        g = torch.Generator()
        g.manual_seed(seed+self.dset)
        
        train_loader = data.DataLoader(dataset, generator=g, **train_loader_kwargs)
        val_loader   = data.DataLoader(dataset, generator=g, **val_loader_kwargs)
        
        return train_loader, val_loader
    
    def _get_data_prefetcher(self, train_loader, val_loader):
        
        train_prefetcher = Data_Prefetcher(train_loader)
        val_prefetcher   = Data_Prefetcher(val_loader)

        del train_loader, val_loader
        torch.cuda.empty_cache()
        return train_prefetcher, val_prefetcher
    
    def _load_one_data(self, train_prefetcher_or_loader, use_data_prefetcher):
        
        print(f'loading 1 / {self.num_train_batches} batch of the dataset...', end=' ')
        start_time = time.time()
        
        if use_data_prefetcher:
            with torch.no_grad():
                x, theta = train_prefetcher_or_loader.next()
        else:
            x, theta = next(iter(train_prefetcher_or_loader))
        
        print(f'takes {time.time() - start_time:.2f} seconds = {(time.time() - start_time) / 60:.2f} minutes')
        print('batch info of the dataset:',
                f'\n| x info:', f'shape {x.shape}, dtype: {x.dtype}, device: {x.device}',
                f'\n| theta info:', f'shape {theta.shape}, dtype: {theta.dtype}, device: {theta.device}',
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
        
        for i in range(4):
            self.posterior_train_set['x'].append(x[i, ...])
            self.posterior_train_set['x_shuffled'].append(x[i, ...][torch.randperm(x.shape[1])])
            self.posterior_train_set['theta'].append(theta[i, ...])
            
            self.posterior_val_set['x'].append(x_val[i, ...])
            self.posterior_val_set['x_shuffled'].append(x_val[i, ...][torch.randperm(x_val.shape[1])])
            self.posterior_val_set['theta'].append(theta_val[i, ...])
        
        # collect of the dataset
        for fig_idx in range(len(self.posterior_train_set['x'])):
            
            figure = plt.figure()
            plt.imshow(self.posterior_train_set['x'][fig_idx][:150, :].cpu())
            plt.savefig(f'{self.log_dir}/posterior/figures/x_train_{fig_idx}_run{self.run}_dset{self.dset}.png')
            self._summary_writer.add_figure(f"data_run{self.run}/x_train_{fig_idx}", figure, self.dset)
            plt.close(figure)
            
            figure = plt.figure()
            plt.imshow(self.posterior_train_set['x_shuffled'][fig_idx][:150, :].cpu())
            plt.savefig(f'{self.log_dir}/posterior/figures/x_train_{fig_idx}_run{self.run}_dset{self.dset}.png')
            self._summary_writer.add_figure(f"data_run{self.run}/x_train_{fig_idx}_shuffled", figure, self.dset)
            plt.close(figure)
        
        
            figure = plt.figure()
            plt.imshow(self.posterior_val_set['x'][fig_idx][:150, :].cpu())
            plt.savefig(f'{self.log_dir}/posterior/figures/x_train_{fig_idx}_run{self.run}_dset{self.dset}.png')
            self._summary_writer.add_figure(f"data_run{self.run}/x_val_{fig_idx}", figure, self.dset)
            plt.close(figure)
            
            figure = plt.figure()
            plt.imshow(self.posterior_val_set['x_shuffled'][fig_idx][:150, :].cpu())
            plt.savefig(f'{self.log_dir}/posterior/figures/x_train_{fig_idx}_run{self.run}_dset{self.dset}.png')
            self._summary_writer.add_figure(f"data_run{self.run}/x_val_{fig_idx}_shuffled", figure, self.dset)
            plt.close(figure)

        print(f'takes {time.time() - start_time:.2f} seconds = {(time.time() - start_time) / 60:.2f} minutes')
    
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
        
        self.optimizer = optim.Adam(list(self._neural_net.parameters()), lr=training_kwargs['learning_rate'])

        if training_kwargs['scheduler'] == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, **training_kwargs['scheduler_params'])
        if training_kwargs['scheduler'] == 'CosineAnnealingWarmRestarts':
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, **training_kwargs['scheduler_params'])
        # if training_kwargs['scheduler'] == 'CosineAnnealingLR':
        #     self.scheduler = CosineAnnealingLR(self.optimizer, *training_kwargs['scheduler_params'])
    
    def _train_one_epoch(self, clip_max_norm, print_freq, train_prefetcher_or_loader, x, theta):
        epoch_start_time = time.time()
        batch_timer      = time.time()
                    
        self.train_data_size    = 0
        train_log_probs_sum     = 0
        train_batch_num         = 0
                    
        # === train network === 
        if self.use_data_prefetcher:
            while x is not None:
                self.optimizer.zero_grad()
                
                # train one batch and log progress
                train_loss, train_log_probs_sum = self._train_one_batch(x, theta, clip_max_norm, train_log_probs_sum)
                self.optimizer.step()
                self._train_one_batch_log(print_freq, batch_timer, train_batch_num, train_loss)
                            
                # get next batch
                train_batch_num += 1
                self.batch_counter += 1
                with torch.no_grad():
                    x, theta = train_prefetcher_or_loader.next()
        else:
            del x, theta
            for x, theta in train_prefetcher_or_loader:
                self.optimizer.zero_grad()
                
                # train one batch and log progress
                train_loss, train_log_probs_sum = self._train_one_batch(x, theta, clip_max_norm, train_log_probs_sum)
                self.optimizer.step()
                self._train_one_batch_log(print_freq, batch_timer, train_batch_num, train_loss)
                            
                # get next batch
                train_batch_num += 1
                self.batch_counter += 1
        
        return epoch_start_time,train_log_probs_sum

    def _train_one_epoch_log(self, train_log_probs_sum):
        
        train_log_prob_average = train_log_probs_sum / self.train_data_size
        # train_log_prob_average = train_log_probs_sum / (self.num_train_batches * dataset_kwargs['batch_size'] * dataset_kwargs['num_probR_sample'])
        self._summary_writer.add_scalars("log_probs", {'training': train_log_prob_average}, self.epoch_counter)
        self._summary_writer.add_scalar("learning rate", self.scheduler.optimizer.param_groups[0]['lr'], self.epoch_counter)
        self._summary["training_log_probs"].append(train_log_prob_average)
        
        return train_log_prob_average
    
    def _train_one_batch(self, x, theta, clip_max_norm, train_log_probs_sum):
        
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
        train_loss = torch.mean(train_losses)
        train_log_probs_sum -= train_losses.sum().item()

        train_loss.backward()
        if clip_max_norm is not None:
            clip_grad_norm_(
                self._neural_net.parameters(), max_norm=clip_max_norm
            )
            
        del x, theta, masks_batch, train_losses
        torch.cuda.empty_cache()
        
        return train_loss, train_log_probs_sum 
    
    def _train_one_batch_log(self, print_freq, batch_timer, train_batch_num, train_loss, do_print_memory_usage=do_print_mem):
        
        if print_freq == 0: # do nothing
            pass

        elif self.num_train_batches <= print_freq:
            print_mem_info('memory usage after batch', do_print_memory_usage)
            print(f'epoch {self.epoch}: batch {train_batch_num} train_loss {-1*train_loss:.2f}, time {(time.time() - batch_timer)/60:.2f}min', end=' ')

        elif train_batch_num % (self.num_train_batches//print_freq) == 0: # print every 5% of batches
            print_mem_info('memory usage after batch', do_print_memory_usage)
            print(f'epoch {self.epoch}: batch {train_batch_num} train_loss {-1*train_loss:.2f}, time {(time.time() - batch_timer)/60:.2f}min', end=' ')

        # self._summary_writer.add_scalar("train_loss_batch", train_loss, self.batch_counter)
    
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
                val_log_prob_sum -= val_losses.sum().item()
                val_data_size    += len(x_val)
                
                del x_val, theta_val, masks_batch
                torch.cuda.empty_cache()
                
                # get next batch
                x_val, theta_val = val_prefetcher_or_loader.next()
                
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
                torch.cuda.empty_cache()
        
        return val_log_prob_sum / val_data_size
    
    def _val_one_epoch_log(self, epoch_start_time):
        
        # print(f'epoch {self.epoch}: val_log_prob {self._val_log_prob:.2f}')
        self._summary_writer.add_scalars("log_probs", {'validation': self._val_log_prob}, self.epoch_counter)
        
        self._summary["validation_log_probs"].append(self._val_log_prob)
        self._summary["epoch_durations_sec"].append(time.time() - epoch_start_time)
    
    def _posterior_behavior_log(self, config, limits):
        
        with torch.no_grad():
            
            epoch = self.epoch
            current_net = deepcopy(self._neural_net)

            # if epoch%config['train']['posterior']['step'] == 0:
            posterior_start_time = time.time()
            print("--> Plotting posterior...", end=" ")

            posterior = self.build_posterior(current_net)
            self._model_bank = [] # clear model bank to avoid memory leak
            
            for fig_idx in range(len(self.posterior_train_set['x'])):
                
                # plot posterior - train x
                fig_x, _ = plot_posterior_with_label(
                    posterior       = posterior, 
                    sample_num      = config['train']['posterior']['sampling_num'],
                    x               = self.posterior_train_set['x'][fig_idx].to(self._device),
                    true_params     = self.posterior_train_set['theta'][fig_idx],
                    limits          = limits,
                    prior_labels    = config['prior']['prior_labels'],
                )
                plt.savefig(f"{self.log_dir}/posterior/figures/posterior_x_train_{fig_idx}_epoch_{self.epoch_counter}.png")
                self._summary_writer.add_figure(f"posterior/x_train_{fig_idx}", fig_x, self.epoch_counter)
                plt.close(fig_x)
                del fig_x, _
                torch.cuda.empty_cache()
                
                # plot posterior - train x_shuffled
                fig_x, _ = plot_posterior_with_label(
                    posterior       = posterior, 
                    sample_num      = config['train']['posterior']['sampling_num'],
                    x               = self.posterior_train_set['x_shuffled'][fig_idx].to(self._device),
                    true_params     = self.posterior_train_set['theta'][fig_idx],
                    limits          = limits,
                    prior_labels    = config['prior']['prior_labels'],
                )
                plt.savefig(f"{self.log_dir}/posterior/figures/posterior_x_train_{fig_idx}_epoch_{self.epoch_counter}.png")
                self._summary_writer.add_figure(f"posterior/x_train_{fig_idx}_shuffled", fig_x, self.epoch_counter)
                plt.close(fig_x)
                del fig_x, _
                torch.cuda.empty_cache()
                
                # plot posterior - val x
                fig_x_val, _ = plot_posterior_with_label(
                    posterior       = posterior, 
                    sample_num      = config['train']['posterior']['sampling_num'],
                    x               = self.posterior_val_set['x'][fig_idx].to(self._device),
                    true_params     = self.posterior_val_set['theta'][fig_idx],
                    limits          = limits,
                    prior_labels    = config['prior']['prior_labels'],
                )
                plt.savefig(f"{self.log_dir}/posterior/figures/posterior_x_val_{fig_idx}_epoch_{self.epoch_counter}.png")
                self._summary_writer.add_figure(f"posterior/x_val_{fig_idx}", fig_x_val, self.epoch_counter)
                plt.close(fig_x_val)
                del fig_x_val, _
                torch.cuda.empty_cache()
                
                # plot posterior - val x_shuffled
                fig_x_val, _ = plot_posterior_with_label(
                    posterior       = posterior, 
                    sample_num      = config['train']['posterior']['sampling_num'],
                    x               = self.posterior_val_set['x_shuffled'][fig_idx].to(self._device),
                    true_params     = self.posterior_val_set['theta'][fig_idx],
                    limits          = limits,
                    prior_labels    = config['prior']['prior_labels'],
                )
                plt.savefig(f"{self.log_dir}/posterior/figures/posterior_x_val_{fig_idx}_epoch_{self.epoch_counter}.png")
                self._summary_writer.add_figure(f"posterior/x_val_{fig_idx}_shuffled", fig_x_val, self.epoch_counter)
                plt.close(fig_x_val)
                del fig_x_val, _
                torch.cuda.empty_cache()
                
            del posterior, current_net
            torch.cuda.empty_cache()
                
            print(f"posterior check finished in {(time.time()-posterior_start_time)/60:.2f}min")
    
    def _converged(self, epoch: int, stop_after_epochs: int, improvement_threshold: float, min_num_epochs: int) -> bool:
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

        # improvement = self._val_log_prob - self._best_val_log_prob
        # (Re)-start the epoch count with the first epoch or any improvement.
        if epoch == 0 or self._val_log_prob > self._best_val_log_prob: # and improvement > improvement_threshold:
            
            self._epochs_since_last_improvement = 0
            
            self._best_val_log_prob     = self._val_log_prob
            self._best_model_state_dict = deepcopy(neural_net.state_dict())
            self._best_model_from_epoch = epoch
            
            if epoch != 0:
                self._posterior_behavior_log(self.config, self.prior_limits) # plot posterior behavior when best model is updated
            # torch.save(deepcopy(neural_net.state_dict()), f"{self.log_dir}/model/best_model_state_dict_run{self.run}.pt")
        
        else:
            self._epochs_since_last_improvement += 1

        # If no validation improvement over many epochs, stop training.
        if self._epochs_since_last_improvement > stop_after_epochs - 1 and epoch > self._epoch_of_last_dset+min_num_epochs:
            # neural_net.load_state_dict(self._best_model_state_dict)
            converged = True
            
            self._neural_net.load_state_dict(self._best_model_state_dict)
            self._val_log_prob = self._best_val_log_prob
            self._epochs_since_last_improvement = 0
            self._epoch_of_last_dset = epoch
            
        return converged
    
    def _converged_dset(self, stop_after_dsets, improvement_threshold, min_num_dsets):
        
        converged = False
        assert self._neural_net is not None
        
        # improvement = self._val_log_prob - self._best_val_log_prob
        if self.dset == 0 or self._val_log_prob_dset > self._best_val_log_prob_dset: # or improvement > improvement_threshold:
            
            self._dset_since_last_improvement = 0
            
            self._best_val_log_prob_dset        = self._val_log_prob_dset
            self._best_model_state_dict_dset    = deepcopy(self._neural_net.state_dict())
            self._best_model_from_dset          = self.dset - 1
            
            torch.save(deepcopy(self._neural_net.state_dict()), f"{self.log_dir}/model/best_model_state_dict_run{self.run}.pt")
        
        else:
            self._dset_since_last_improvement += 1
            
        if self._dset_since_last_improvement > stop_after_dsets - 1 and self.dset > min_num_dsets - 1:
            
            converged = True
            
            self._neural_net.load_state_dict(self._best_model_state_dict_dset)
            self._val_log_prob_dset = self._best_val_log_prob_dset
            self._dset_since_last_improvement = 0
            
            torch.save(deepcopy(self._neural_net.state_dict()), f"{self.log_dir}/model/best_model_state_dict_run{self.run}.pt")
        
        return converged
    
    @staticmethod
    def _show_epoch_progress(epoch, starting_time, train_log_prob, val_log_prob):
        print(f"\n| Epochs trained: {epoch:5}. Time elapsed {(time.time()-starting_time)/ 60:6.2f}min ||  log_prob train: {train_log_prob:.2f} | log_prob val: {val_log_prob:.2f}")
    
    def _describe_dset(self) -> str:
        
        return f"""
        -------------------------
        ||||| RUN {self.run} dset {self.dset} STATS |||||:
        -------------------------
        Total epochs trained: {self.epoch_counter}
        Best validation performance: {self._best_val_log_prob:.4f}, from epoch {self._best_model_from_epoch:5}
        Model from best epoch {self._best_model_from_epoch} is loaded for further training
        -------------------------
        """
    
    
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
        log_dir,
        config,
        
        seed,
        prior_limits,
        
        dataset_kwargs,
        dataloader_kwargs,
        training_kwargs,
        continue_from_checkpoint=None,
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
        self._num_atoms = training_kwargs["num_atoms"]
        self._use_combined_loss = training_kwargs["use_combined_loss"]
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
    

