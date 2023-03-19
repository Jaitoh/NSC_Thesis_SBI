'''
train with conventional features (mel-spectrogram)
'''

debug=False

import sys
if debug:
    # sys.path.append('./src') # src
    sys.path.append('./simulation/src') # src
else:
    sys.path.append('../../') # src

from models.networks import ASR_Model
# from data.datasets import TIDIGITS_ASR_FEATURE
from utils import is_file_in, calculate_accuracy, list_TIDIGITS_files
from data.features import Noisy_Features
from data.datasets import GSC_DATASET, TIDIGITS_DC_DATASET

import os
import csv
import argparse
import yaml
import h5py
import time
import shutil
import hiyapyco
from pathlib import Path
import glob
from tqdm import tqdm

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchaudio

import matplotlib.pyplot as plt
import joblib

if debug:
    sim_folder_dir = Path('./')
    sim_folder_dir = Path('./simulation') 
else:
    sim_folder_dir = Path('../../../') # /simulation/src/trains/noise
    
# print(glob.glob(f"{sim_folder_dir}/src/trains/noise/config_/*.yaml"))
# audio_wav_dir = sim_folder_dir / 'datasets' / 'GSC_v2' / 'audio'
# dataset_dir = Path('/mnt/scratch/wenjie/')
# audio_wav_dir = dataset_dir / 'datasets' / 'GSC_v2' / 'audio'
# audio_wav_dir = sim_folder_dir / 'datasets' / 'GSC_v2' / 'audio'

## ============================================================ set seed ===
def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

## ============================================================ get args ===
def get_args():
    parser = argparse.ArgumentParser(description='Run DC')
    parser.add_argument('--run_test', action='store_true', help='test mode')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--config_sp_dir', type=str, default=f"{sim_folder_dir}/src/trains/noise/config/sp1.yaml",  help='speech configuration file directory')
    parser.add_argument('--config_n_dir', type=str, default=f"{sim_folder_dir}/src/trains/noise/config/n0.yaml",  help='noise configuration file directory')
    parser.add_argument('--config_f_dir', type=str, default=f"{sim_folder_dir}/src/trains/noise/config/f1p0.yaml",  help='feature configuration file directory')
    parser.add_argument('--config_m_dir', type=str, default=f"{sim_folder_dir}/src/trains/noise/config/m0p0.yaml", help='model configuration file directory')
    # parser.add_argument('--f_dir', type=str, default=f"{sim_folder_dir}/datasets/TIDIGITS/simulated/", help='output feature file directory')
    # parser.add_argument('--f_dir', type=str, default=f"{sim_folder_dir}/datasets/GSC_v2/simulated/", help='output feature file directory')
    # parser.add_argument('--f_dir', type=str, default=f"/mnt/scratch/wenjie/datasets/GSC_v2/simulated/", help='output feature file directory')
    if debug:
        parser.add_argument('--log_dir', type=str, default=f"{sim_folder_dir}/src/trains/noise/logs/run_test/sp1n0_f1p0_m0p0_s0_test", help='run recoding folder: e.g. n0_f0p0_m0p0_s0_test')
    else:
        parser.add_argument('--log_dir', type=str, default=f"{sim_folder_dir}/src/trains/noise/logs/run_test/sp1n0_f1p0_m0p0_s0", help='run recoding folder: e.g. n0_f0p0_m0p0_s0')
    args = parser.parse_args()
    return args

## ============================================================ trainer ===
class Trainer():
    
    def __init__(self):
        
        self.args       = get_args()
        print(self.args)
        self.log_dir    = self.args.log_dir
        self.config     = self._merge_config()
        print(f'configuration: \n{self.config}')

        # model parameters
        self.opt_type        = self.config['model']['opt_type']
        self.scheduler_type  = self.config['model']['scheduler_type']
        self.scheduler_mode  = self.config['model']['scheduler_mode']

        if self.args.run_test or debug:
            self.num_epochs  = 2
        else:
            self.num_epochs  = self.config['model']['num_epochs']
        self.batch_size      = self.config['model']['batch_size']
        self.learning_rate   = self.config['model']['learning_rate']
        self.clip_grad       = self.config['model']['clip_grad']
        
        ## === set seed ===
        seed = self.args.seed
        setup_seed(seed)
        self.g = torch.Generator()
        self.g.manual_seed(seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'device: {self.device}')
        
        self.speech_dir = self.config['speech']['speech_dir']
        self.speech_type = self.config['speech']['type']
        self.speech_id = self.args.config_sp_dir.split('/')[-1].split('.')[0]
        self.noise_id   = self.args.config_n_dir.split('/')[-1].split('.')[0]
        self.feature_id = self.args.config_f_dir.split('/')[-1].split('.')[0]
        # ============================== preprocess and save data ==========
        # if the f_dir is in the folder, skip preprocessing
        if debug or self.args.run_test:
            self.snr_list = [50, 0]
        else:   
            self.snr_list = [ 50, 20, 10, 0, -5, -10 ] #[-10, -5, 0, 10, 50]
            self.snr_list = [ 50, 10, 0, -10 ] #[-10, -5, 0, 10, 50]
        
        for snr in self.snr_list:
            if not is_file_in(self.speech_dir + f'/simulated/{self.speech_id}{self.noise_id}_{self.feature_id}_snr{snr}.hdf5'):
                self.preprocess_save_data(snr)
            else:
                print(f'data file {self.speech_dir}/simulated/{self.noise_id}_{self.feature_id}_snr{snr}.hdf5 exists, skip preprocessing...')
        
    def _merge_config(self):
        # config_ = yaml.load(config_dir, Loader=yaml.FullLoader)
        # create log_dir when it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        config = hiyapyco.load([self.args.config_sp_dir, self.args.config_n_dir, self.args.config_f_dir, self.args.config_m_dir], method=hiyapyco.METHOD_MERGE)
        shutil.copyfile(self.args.config_sp_dir, Path(self.log_dir) / 'config_sp.yaml')
        shutil.copyfile(self.args.config_n_dir, Path(self.log_dir) / 'config_n.yaml')
        shutil.copyfile(self.args.config_f_dir, Path(self.log_dir) / 'config_f.yaml')
        shutil.copyfile(self.args.config_m_dir, Path(self.log_dir) / 'config_m.yaml')
        
        return config
    
    def _get_noisy_feature(self, signal_dir, snr, seed):
        
        Speech_Noise =  Noisy_Features( signal_dir, 
                                        sr_new       =self.config['feature']['params']['sr'], 
                                        noise_type   =self.config['noise']['type'],
                                        snr          =snr,
                                        babble_torch =self.babble_torch,
                                        babble_sr    =self.babble_sr,
                                        seed = seed)
        
        if self.config['feature']['name'].lower() == 'mels':
            feature = Speech_Noise.generate_melS(config=self.config['feature']['params']).T # (T, C)
        elif self.config['feature']['name'].lower() == 'ppfm':
            feature = Speech_Noise.generate_event_frame(config=self.config['feature']['params']) #[T, C]
        else:
            raise ValueError(f"feature {self.config['feature']['name']} is not implemented")
        
        return feature
    
    def preprocess_save_data(self, snr):
        
        speech_id, noise_id, feature_id = self.speech_id, self.noise_id, self.feature_id
        
        # if self.config_['noise']['type'] == 'babble':
        babble_torch, self.babble_sr = torchaudio.load(self.config['noise']['babble_dir'])
        self.babble_torch = babble_torch[0]
        
        if self.speech_type == 'GSC':
            with open(f"{self.speech_dir}/audio/training_list.txt", 'r') as f:
                train_file = f.read().splitlines()
            with open(f"{self.speech_dir}/audio/validation_list.txt", 'r') as f:
                valid_file = f.read().splitlines()
            with open(f"{self.speech_dir}/audio/testing_list.txt", 'r') as f:
                test_file = f.read().splitlines()
            train_files = [f"{self.speech_dir}/audio/{i}" for i in train_file]
            valid_files = [f"{self.speech_dir}/audio/{i}" for i in valid_file]
            test_files  = [f"{self.speech_dir}/audio/{i}" for i in test_file]
            
            train_info = [','.join((i.split('.')[0].split('/')[0], i.split('.')[0].split('/')[1])) for i in train_file]
            valid_info = [','.join((i.split('.')[0].split('/')[0], i.split('.')[0].split('/')[1])) for i in valid_file]
            test_info  = [','.join((i.split('.')[0].split('/')[0], i.split('.')[0].split('/')[1])) for i in test_file]
            
        if self.speech_type == 'TIDIGITS': # train files are full path, different from GSC
            train_files, train_info, test_files, test_info = list_TIDIGITS_files(f'{self.speech_dir}/audio', is_digit=True)
            valid_files, valid_info = test_files, test_info
            
        # create a feature file
        f = h5py.File(self.speech_dir + f'/simulated/{speech_id}{noise_id}_{feature_id}_snr{snr}.hdf5', 'w')
        
        if debug or self.args.run_test:
            train_files = train_files[:40]
            valid_files = valid_files[:40]
            test_files  = test_files [:40]
            # f.create_dataset("train_info", data=train_info[:40])
            # f.create_dataset("valid_info", data=valid_info[:40])
            # f.create_dataset("test_info" , data=test_info [:40])
            len_train_files = len(train_files[:40])
            len_valid_files = len(valid_files[:40])
            len_test_files  = len(test_files [:40])
        else:
            # f.create_dataset("train_info", data=train_info)
            # f.create_dataset("valid_info", data=valid_info)
            # f.create_dataset("test_info" , data=test_info )
            len_train_files = len(train_files)
            len_valid_files = len(valid_files)
            len_test_files  = len(test_files)
        
        # ======================================== generate train features ===
        train_f = f.create_group('train_feature')
        print(f"converting train features {self.config['feature']['name']}:  '{speech_id}{noise_id}_{feature_id}_snr{snr}.hdf5'...")
        
        # create folder if not exist for saving the features
        if not os.path.exists(f'{self.log_dir}/feature'):
            os.makedirs(f'{self.log_dir}/feature')
            
        for i in tqdm(range(len(train_files))):
        # for i in range(1):
            
            # if (i) % 1000 == 0:
            #     print(f'Writing [{i+1:5}/{len(train_files):5}] Elapsed [{time.time()-start_time:.2f}s]')
            signal_dir = train_files[i]
            feature = self._get_noisy_feature(signal_dir, snr, seed=i+0)
            
            info = f'{train_info[i]},snr{snr}'
            train_f.create_dataset(info, data=feature)
        
        fig=plt.pcolormesh(feature.T) 
        plt.savefig(f"{self.log_dir}/feature/train_feature_{speech_id}{noise_id}_{feature_id}_snr{snr}.png")
        
        # ====================================== generate validate features ===
        valid_f = f.create_group('valid_feature')
        print(f"converting valid features {self.config['feature']['name']}:  '{speech_id}{noise_id}_{feature_id}_snr{snr}.hdf5'...")
        
        for i in tqdm(range(len(valid_files))):
        # for i in range(1):
            
            # if (i) % 1000 == 0:
            #     print(f'Writing [{i+1:5}/{len(valid_files):5}] Elapsed [{time.time()-start_time:.2f}s]')
            signal_dir = valid_files[i]
            feature = self._get_noisy_feature(signal_dir, snr, seed=i+len_train_files)
                
            info =  f'{valid_info[i]},snr{snr}'
            valid_f.create_dataset(info, data=feature)
        
        fig=plt.pcolormesh(feature.T)
        plt.savefig(f"{self.log_dir}/feature/valid_feature_{speech_id}{noise_id}_{feature_id}_snr{snr}.png")
        
        # ========================================== generate test features ===
        test_f = f.create_group('test_feature')
        print(f"converting test features {self.config['feature']['name']}:  '{speech_id}{noise_id}_{feature_id}_snr{snr}.hdf5'...")
        
        for i in tqdm(range(len(test_files))):
        # for i in range(1):
            
            # if (i) % 1000 == 0:
            #     print(f'Writing [{i+1:5}/{len(test_files):5}] Elapsed [{time.time()-start_time:.2f}s]')
            
            signal_dir = test_files[i]
            feature = self._get_noisy_feature(signal_dir, snr, seed=i+len_train_files+len_valid_files)
            
            info =  f'{test_info[i]},snr{snr}'
            test_f.create_dataset(info, data=feature)
        
        fig=plt.pcolormesh(feature.T) 
        plt.savefig(f"{self.log_dir}/feature/test_feature_{noise_id}_{feature_id}_snr{snr}.png")
        
        f.close()
        
    def _load_model_settings(self, train_loader):
        n_class = 35 if self.speech_type == 'GSC' else 11
        self.model = ASR_Model(input_size=self.data_shape[-1], n_out_class=n_class).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = self._get_optimizer()
        if self.scheduler_type!='none':
            self.scheduler = self._get_scheduler(train_loader)
    
    def _reload_optimizer_scheduler(self, train_loader):
        self.optimizer = self._get_optimizer()
        if self.scheduler_type!='none':
            self.scheduler = self._get_scheduler(train_loader)
        
    def _get_optimizer(self):
        if self.opt_type == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.config['model']['weight_decay'])
        if self.opt_type == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.config['model']['weight_decay'])
        return optimizer

    def _get_scheduler(self, train_loader):
        if self.scheduler_type == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, 
                                                            max_lr  = self.learning_rate,
                                                            steps_per_epoch=len(train_loader),
                                                            epochs=self.num_epochs,
                                                            anneal_strategy='linear')
            self.scheduler_mode = 'batch'
            
        if self.scheduler_type == 'cosine':
            if self.scheduler_mode == 'epoch':
                # steps = int(self.num_epochs//3) if self.num_epochs >= 3 else 1
                steps = int(self.num_epochs)
            if self.scheduler_mode == 'batch':
                steps = int(len(train_loader))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=steps, eta_min=1e-6)
            
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0)
        # scheduler_mode = 'batch'
        return scheduler
        
    def _load_all_dataset(self):
        """ load all datasets to memory
        """
        # self.all_train_dataset, self.all_valid_dataset, self.all_test_dataset = [], [], []
        self.all_train_loader , self.all_valid_loader , self.all_test_loader  = [], [], []
        batch_size    = self.config['model']['batch_size']
        
        for i in tqdm(range(len(self.snr_list))):
            
            norm_type = self.config['model']['norm_type']
            
            dataset_dir_list = []
            for j in range(i+1):
                snr = self.snr_list[j]
                dataset_dir = self.speech_dir + f'/simulated/{self.speech_id}{self.noise_id}_{self.feature_id}_snr{snr}.hdf5'
                dataset_dir_list.append(dataset_dir)
            
            dataset_dir_test = [self.speech_dir + f'/simulated/{self.speech_id}{self.noise_id}_{self.feature_id}_snr{self.snr_list[i]}.hdf5']
            
            if self.speech_type == 'GSC':
                if debug or self.args.run_test: # use small test dataset to save loading time
                    train_dataset = GSC_DATASET(dataset_dir_list, 'test', norm_type)
                    valid_dataset = GSC_DATASET(dataset_dir_list, 'test', norm_type)
                    test_dataset  = GSC_DATASET(dataset_dir_test, 'test', norm_type)
                else:    
                    train_dataset = GSC_DATASET(dataset_dir_list, 'train', norm_type)
                    valid_dataset = GSC_DATASET(dataset_dir_list, 'valid', norm_type)
                    test_dataset  = GSC_DATASET(dataset_dir_test, 'test' , norm_type)
            else:
                if debug or self.args.run_test: # use small test dataset to save loading time
                    train_dataset = TIDIGITS_DC_DATASET(dataset_dir_list, 'test', norm_type)
                    valid_dataset = TIDIGITS_DC_DATASET(dataset_dir_list, 'test', norm_type)
                    test_dataset  = TIDIGITS_DC_DATASET(dataset_dir_test, 'test', norm_type)
                else:    
                    train_dataset = TIDIGITS_DC_DATASET(dataset_dir_list, 'train', norm_type)
                    valid_dataset = TIDIGITS_DC_DATASET(dataset_dir_list, 'valid', norm_type)
                    test_dataset  = TIDIGITS_DC_DATASET(dataset_dir_test, 'test' , norm_type)
                
            
            for idx in np.random.randint(0, len(train_dataset), 2):
                fig=plt.pcolormesh(train_dataset[idx][0].T)
                plt.colorbar(fig)
                plt.savefig(f"{self.log_dir}/feature/dataset_normalized_train_feature_snr{i}_{snr}_{idx}.png")
                plt.close()
            
            for idx in np.random.randint(0, len(valid_dataset), 2):
                fig=plt.pcolormesh(valid_dataset[idx][0].T)
                plt.colorbar(fig)
                plt.savefig(f"{self.log_dir}/feature/dataset_normalized_valid_feature_snr{i}_{snr}_{idx}.png")
                plt.close()
                
            for idx in np.random.randint(0, len(test_dataset), 2):
                fig=plt.pcolormesh(test_dataset[idx][0].T)
                plt.colorbar(fig)
                plt.savefig(f"{self.log_dir}/feature/dataset_normalized_test_feature_snr{i}_{snr}_{idx}.png")
                plt.close()

            train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, worker_init_fn=seed_worker, generator=self.g,
                                    collate_fn=train_dataset.my_collate(), drop_last=True, pin_memory=True)
            valid_loader  = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=16, worker_init_fn=seed_worker, generator=self.g,
                                    collate_fn=valid_dataset.my_collate(), drop_last=False, pin_memory=True)
            test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=16, worker_init_fn=seed_worker, generator=self.g,
                                    collate_fn=test_dataset.my_collate(), drop_last=False, pin_memory=True)
            
            self.all_train_loader.append(train_loader)
            self.all_valid_loader.append(valid_loader)
            self.all_test_loader.append(test_loader)
        
        self.data_shape = train_dataset[0][0].shape
        print(f'data shape: {self.data_shape}') # [T,C]
        
    # def _dataset_train_loader(self, snr):
    #     """ data loaders
    #     """
    #     # tic = time.time()
    #     batch_size    = self.config_['model']['batch_size']
    #     # print(f'loading train data {self.speech_dir + f'{self.noise_id}_{self.feature_id}_snr{snr}.hdf5'}')
    #     if debug or self.args.run_test: # use test dataset to save loading time
    #         self.train_dataset = GSC_MELS_DATASET(self.speech_dir + f'{self.noise_id}_{self.feature_id}_snr{snr}.hdf5',
    #                                     train_valid_or_test='test',
    #                                     norm_type=self.config_['model']['norm_type'])
        
    #         self.valid_dataset = GSC_MELS_DATASET(self.speech_dir + f'{self.noise_id}_{self.feature_id}_snr{snr}.hdf5', 
    #                                         train_valid_or_test='test',
    #                                         norm_type=self.config_['model']['norm_type'])
    #     else:    
    #         self.train_dataset = GSC_MELS_DATASET(self.speech_dir + f'{self.noise_id}_{self.feature_id}_snr{snr}.hdf5',
    #                                         train_valid_or_test='train',
    #                                         norm_type=self.config_['model']['norm_type'])
            
    #         self.valid_dataset = GSC_MELS_DATASET(self.speech_dir + f'{self.noise_id}_{self.feature_id}_snr{snr}.hdf5', 
    #                                         train_valid_or_test='valid',
    #                                         norm_type=self.config_['model']['norm_type'])
        
    #     for idx in np.random.randint(0, len(self.train_dataset), 10):
    #         fig=plt.pcolormesh(self.train_dataset[idx][0].T)
    #         plt.colorbar(fig)
    #         plt.savefig(f"{self.log_dir}/feature/dataset_normalized_train_feature_{snr}_{idx}.png")
    #         plt.close()
        
    #     for idx in np.random.randint(0, len(self.valid_dataset), 10):
    #         fig=plt.pcolormesh(self.valid_dataset[idx][0].T)
    #         plt.colorbar(fig)
    #         plt.savefig(f"{self.log_dir}/feature/dataset_normalized_valid_feature_{snr}_{idx}.png")
    #         plt.close()
            
    #     self.data_shape = self.train_dataset[0][0].shape
    #     print(f'data shape: {self.data_shape}') # [T,C]
        
    #     train_loader  = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, worker_init_fn=seed_worker, generator=self.g,
    #                             collate_fn=self.train_dataset.my_collate(), drop_last=True, pin_memory=True)
    #     valid_loader  = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=True, num_workers=16, worker_init_fn=seed_worker, generator=self.g,
    #                             collate_fn=self.valid_dataset.my_collate(), drop_last=False, pin_memory=True)
    #     # toc = time.time()
    #     # print(f'loading training data time: {toc-tic:.2f}s')
    
    # def _dataset_load_all_test_dataset(self):
        
    #     self.test_dataset_all, self.test_loader_all = [], []
        
    #     for j, snr in enumerate(self.snr_list):
    #         batch_size    = self.config_['model']['batch_size']
        
    #         test_dataset  = GSC_MELS_DATASET(self.speech_dir + f'{self.noise_id}_{self.feature_id}_snr{snr}.hdf5', 
    #                                         train_valid_or_test='test',
    #                                         norm_type=self.config_['model']['norm_type'])
            
    #         for idx in np.random.randint(0, len(test_dataset), 10):
    #             # idx = 2659
    #             fig=plt.pcolormesh(test_dataset[idx][0].T)
    #             plt.colorbar(fig)
    #             plt.savefig(f"{self.log_dir}/feature/dataset_normalized_test_feature_{snr}_{idx}.png")
    #             plt.close()
                
    #         test_loader   = DataLoader(test_dataset,  batch_size=batch_size, shuffle=True, num_workers=16, worker_init_fn=seed_worker, generator=self.g,
    #                                 collate_fn=test_dataset.my_collate(), drop_last=False, pin_memory=True)
            
    #         self.test_dataset_all.append(test_dataset)
    #         self.test_loader_all.append(test_loader)
        
    # def _get_loader(self, snr, train_valid_or_test):
    #     pass
    # def _dataset_test_loader(self, snr):
    #     """ data loaders
    #     """
    #     # tic = time.time()
    #     batch_size    = self.config_['model']['batch_size']
        
    #     self.test_dataset  = GSC_MELS_DATASET(self.speech_dir + f'{self.noise_id}_{self.feature_id}_snr{snr}.hdf5', 
    #                                     train_valid_or_test='test',
    #                                     norm_type=self.config_['model']['norm_type'])

    #     fig=plt.pcolormesh(self.test_dataset[0][0].T)
    #     plt.savefig(f"{self.log_dir}/dataset_normalized_test_feature_{snr}.png")
        
    #     self.test_loader   = DataLoader(self.test_dataset,  batch_size=batch_size, shuffle=True, num_workers=16, worker_init_fn=seed_worker, generator=self.g,
    #                             collate_fn=self.test_dataset.my_collate(), drop_last=False, pin_memory=True)
        # toc = time.time()
        # print(f'loading test data time: {toc-tic:.2f}s')
    def _feature_writer(self, snr, train_loader):
        
        # self._dataset_train_loader(snr)
        dataiter = iter(train_loader)
        images, labels = next(dataiter)
        img_grid = torchvision.utils.make_grid(images)
        # plt.imshow(img_grid.numpy().transpose((1, 2, 0)))
        # expand one dimension from [N, C, T] to [N, 3, C, T]
        img_grid = img_grid.unsqueeze(1).expand(-1, 3, -1, -1)
        self.writer.add_images(f'train_feature_snr_{snr}', img_grid)
        
        # dataiter = iter(valid_loader)
        # images, labels = next(dataiter)
        # img_grid = torchvision.utils.make_grid(images)
        # # plt.imshow(img_grid.numpy().transpose((1, 2, 0)))
        # self.writer.add_images(f'valid_feature_snr_{snr}', img_grid.numpy(), 0)
        
        # self._dataset_test_loader(snr)
        # dataiter = iter(self.test_loader)
        # images, labels = next(dataiter)
        # img_grid = torchvision.utils.make_grid(images)
        # # plt.imshow(img_grid.numpy().transpose((1, 2, 0)))
        # self.writer.add_image(f'test_feature_snr_{snr}', img_grid)
        
        
    def train_and_test(self):
        # create a csv for logging
        log_header1 = ['Wall time', 'Step', 'Value', 'train SNR']
        log_header2 = ['Wall time', 'Step', 'Value', 'train SNR', 'test SNR']
        # log_train_loss, log_train_acc   = [log_header1], [log_header1]
        # log_valid_loss, log_valid_acc   = [log_header1], [log_header1]
        # log_test_loss , log_test_acc    = [[log_header2] for snr in self.snr_list], [[log_header2] for snr in self.snr_list]
        # log_lr = []
        log_train_loss, log_train_acc   = [], []
        log_valid_loss, log_valid_acc   = [], []
        log_test_loss , log_test_acc    = [[] for snr in self.snr_list], [[] for snr in self.snr_list]
        log_lr = []
        
        # === do train and test ===
        best_loss = float('inf')
        start_time = time.time()
        
        print(f'loading all dataset')
        # tic = time.time()
        self._load_all_dataset() # load all datasets and data loaders
        # toc = time.time()
        # print(f'data loaded {toc-tic:.5f}')
            
        self.train_epochs = 0
        for k, snr in enumerate(self.snr_list):
            print(f'\n============================== train with snr {snr} ==========')
            
            # tic = time.time()
            # self._feature_writer(snr) # write feature to tensorboard
            # toc = time.time()
            # print(f'feature writed to tensorboard {toc-tic:.5f}')
            
            train_loader, valid_loader = self.all_train_loader[k], self.all_valid_loader[k]
            
            if snr == self.snr_list[0]: # train with the first snr
                self._load_model_settings(train_loader) 
                self.writer = SummaryWriter(self.log_dir)
                self.writer.add_graph(model = self.model, input_to_model=torch.rand(1, self.data_shape[0], self.data_shape[1]).to(self.device))
            else:
                self._reload_optimizer_scheduler(train_loader) # reload optimizer and scheduler
            
            for epoch in range(self.num_epochs):
                self.train_epochs += 1
                train_loss, train_acc, valid_loss, valid_acc, lr = self._train(train_loader, valid_loader)
                
                self.writer.add_scalar('train/epoch/loss', train_loss, self.train_epochs)
                self.writer.add_scalar('train/epoch/acc', train_acc, self.train_epochs)
                self.writer.add_scalar('train/epoch/lr', lr, self.train_epochs)
                self.writer.add_scalar('train/epoch/snr', snr, self.train_epochs)
                self.writer.add_scalar('valid/epoch/loss', valid_loss, self.train_epochs)
                self.writer.add_scalar('valid/epoch/acc', valid_acc, self.train_epochs)
                
                # save the model
                if train_loss < best_loss:
                    best_loss = train_loss
                    torch.save(self.model.state_dict(), self.log_dir + '/best_model.pt')

                log_train_loss.append([time.time()-start_time, self.train_epochs, train_loss, snr])
                log_train_acc.append([time.time()-start_time, self.train_epochs, train_acc, snr ])
                
                log_valid_loss.append([time.time()-start_time, self.train_epochs, valid_loss, snr])
                log_valid_acc.append([time.time()-start_time, self.train_epochs, valid_acc, snr ])
                
                if epoch % 1 == 0: # run test for every x epochs
                    # for snr in self.snr_list:
                    for i, snr_test in enumerate(self.snr_list):
                        # === test ===
                        # self._dataset_test_loader(snr_test)
                        # test_loader = self.all_test_loader[i]
                        test_loss, test_acc = self._test(i)
                        
                        self.writer.add_scalar(f'test/epoch/snr{snr_test}/loss', test_loss, self.train_epochs)
                        self.writer.add_scalar(f'test/epoch/snr{snr_test}/acc' , test_acc , self.train_epochs)

                        # self.writer.add_hparams({'lr': self.learning_rate, 'bsize': self.batch_size, 'feature': self.feature},
                        #                         {'hparam/test_loss': test_loss, 'hparam/test_acc': test_acc, 'hparam/test_per': test_per},)
                        log_test_loss[i].append([time.time()-start_time, self.train_epochs, test_loss, snr, snr_test])
                        log_test_acc[i].append([time.time()-start_time, self.train_epochs, test_acc, snr, snr_test])
            
        self.writer.flush()

        # write data into csv
        # log_dir = Path(self.args.log_dir)
        # files = ['train_loss.csv', 'train_acc.csv','valid_loss.csv', 'valid_acc.csv', 'test_loss.csv', 'test_acc.csv']
        # csv_data = [log_train_loss, log_train_acc, log_valid_loss, log_valid_acc, log_test_loss, log_test_acc]
        # for i in range(len(files)):
        #     with open(log_dir/files[i], 'w', encoding='UTF8', newline='') as f:
        #         writer = csv.writer(f)
        #         writer.writerows(csv_data[i])
        
        # create a hdf5 file for logging
        # dt = h5py.special_dtype(vlen=str)
        log_file = h5py.File(f'{self.log_dir}/log_loss_acc.hdf5', 'w')
        log_file.create_dataset('info: train_valid data stored as Wall time, Step, Value, train SNR', data=np.array([0]))
        log_file.create_dataset('info: test data stored as Wall time, Step, Value, train SNR, test SNR', data=np.array([0]))
        log_file.create_dataset('train_loss', data=np.array(log_train_loss))
        log_file.create_dataset('train_acc' , data=np.array(log_train_acc ))
        log_file.create_dataset('valid_loss', data=np.array(log_valid_loss))
        log_file.create_dataset('valid_acc' , data=np.array(log_valid_acc ))
        log_file.create_dataset('test_loss' , data=np.array(log_test_loss ))
        log_file.create_dataset('test_acc'  , data=np.array(log_test_acc  ))
        log_file.close()

        if debug or self.args.run_test:
            # remove one file
            for snr in self.snr_list:
                os.remove(self.speech_dir + f'/simulated/{self.speech_id}{self.noise_id}_{self.feature_id}_snr{snr}.hdf5')
            
            
    def _train(self, train_loader, valid_loader):
        # train the model (no validation) and save the best model
        self.model.train()  
        data_len = len(train_loader.dataset)
        batch_size = train_loader.batch_size
        
        accs, losses = [], []
        lrs = []
        start_time = time.time()
        
        for batch_idx,  (inputs, labels) in enumerate(train_loader):
            # self.counter_train += 1
            
            # inputs, labels = _data # input [B,T,C], label [B,L]
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs.float())  # output [B,T,C]
            # outputs = outputs[:,-1,:] # take the last timestemp output as final output
            outputs = torch.max(outputs, dim=1).values # [B,C] # take max of each class of the entire sequence as output
            
            loss = self.criterion(outputs, labels) 
            loss.backward()
            
            if self.clip_grad!=0:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad, norm_type=2)
            
            self.optimizer.step()
            
            if self.scheduler_mode=='batch':
                self.scheduler.step()
                
            with torch.no_grad():
                acc = calculate_accuracy(outputs, labels)
                accs.append(acc.item())
                losses.append(loss.item())

                # self.print_batch_train(epoch, batch_idx, batch_size, data_len, loss, ler, per)
                
                # self.writer.add_scalar('train/batch/loss', loss, self.counter_train)
                # self.writer.add_scalar('train/batch/acc', acc, self.counter_train)
                # self.writer.add_scalar('train/batch/lr', self.scheduler.get_last_lr()[0], self.counter_train)
                # writer.add_histogram("fc1", model_d.fc1.weight)
                # writer.add_histogram("fc2", model_d.fc2.weight)
                
        if self.scheduler_mode=='epoch':
            self.scheduler.step()
            # lrs.append(scheduler.get_last_lr()[0])
            
        with torch.no_grad():
            
            if self.scheduler_mode!='none':
                lr = self.scheduler.get_last_lr()[0]
            else:
                lr = self.optimizer.param_groups[0]['lr']
                
            train_loss, train_acc = np.mean(losses), np.mean(accs)
            duration = time.time() - start_time
            
            self.print_epoch_train(self.train_epochs, data_len, train_loss, train_acc, duration, lr)

            if self.speech_type == 'TIDIGITS':
                valid_loss, valid_acc = 0, 0 # no validation data for TIDIGITS
            else:
                valid_loss, valid_acc =self._valid(self.train_epochs, valid_loader)
            
        return train_loss, train_acc, valid_loss, valid_acc, lr
    
    def _valid(self, epoch, valid_loader):
        # validation
        self.model.eval()
        data_len = len(valid_loader.dataset)
        batch_size = valid_loader.batch_size
        
        accs, losses = [], []
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, _data in enumerate(valid_loader):
                
                inputs, labels = _data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs.float())  # output [B,T,C]
                outputs = torch.max(outputs, dim=1).values # [B,C] # take max of each class of the entire sequence as output
                
                loss = self.criterion(outputs, labels)
                losses.append(loss.item()*inputs.shape[0]) 
                
                acc = calculate_accuracy(outputs, labels)
                accs.append(acc.item()*inputs.shape[0])
            
            valid_acc  = np.sum(accs)/data_len
            valid_loss = np.sum(losses)/data_len
            duration = time.time() - start_time
            
            self.print_epoch_valid(epoch, data_len, valid_loss, valid_acc, duration)
            
            return valid_loss, valid_acc
                
    def _test(self, idx):
        self.model.eval()
        
        test_loader = self.all_test_loader[idx]
        data_len = len(test_loader.dataset)
        test_accs, test_losses = [], []
        
        with torch.no_grad():
            start_time = time.time()
            for i, _data in enumerate(test_loader):
                # self.counter_test += 1
                
                inputs, labels = _data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs.float())
                # outputs = outputs[:,-1,:] # take the last timestemp output as final output
                outputs = torch.max(outputs, dim=1).values # [B,C] # take max of each class of the entire sequence as output
            
                loss = self.criterion(outputs, labels)
                test_losses.append(loss.item()*inputs.shape[0]) 
                
                acc = calculate_accuracy(outputs, labels)
                test_accs.append(acc.item()*inputs.shape[0])
                
                # self.print_batch_test(epoch, i, batch_size, data_len, loss, output_str, labels_str)
                # self.writer.add_scalar('test/batch/loss', loss, self.counter_test)
                # self.writer.add_scalar('test/batch/ler', ler, self.counter_test)
                # self.writer.add_scalar('test/batch/per', per, self.counter_test)
                
            test_acc  = np.sum(test_accs)/data_len
            test_loss = np.sum(test_losses)/data_len
            duration = time.time() - start_time
            
            self.print_epoch_test(self.snr_list[idx], data_len, test_loss, test_acc, duration)
            
            return test_loss, test_acc
        
    @staticmethod
    def print_batch_train(epoch, batch_idx, batch_size, data_len, loss, ler, per):
        if (batch_idx+1) % 100 == 0 or (batch_idx+1)*batch_size >= data_len-batch_size: # due to drop last
            if (batch_idx+1)*batch_size >= data_len-batch_size: # last batch
                data_count = data_len
                percent = 100
                ending = '\n'
            else:
                data_count = (batch_idx+1)*batch_size
                percent = data_count / data_len * 100
                ending = '\r'
            print(f' -- Train Epoch: {epoch:3} [{data_count:5}/{data_len} ({percent:3.0f}%)]\tLoss: {loss.item():.6f} LER {ler:3.3f} PER {per:3.3f} ', end=ending)
    
    @staticmethod
    def print_epoch_train(epoch, data_len, train_loss, train_acc, duration, lr):
            # print(f'Train Epoch: {epoch:3} [{data_len:5}/{data_len} ({100:3.0f}%)]\tLoss: {train_loss:8.5f} LER {avg_ler:6.3f} PER {avg_per:6.3f} \tpred: {output_str[0]}, target: {labels_str[0]}', ending = '\r')
            print(f'Train Epoch: {epoch:3} {data_len:5}  Loss:[{train_loss:8.5f}]  ACC:[{train_acc*100:7.3f}%], time: ({duration:5.2f}), lr: {lr:.5f}', end=' ')
    
    @staticmethod
    def print_batch_test(epoch, i, batch_size, data_len, loss, output_str, labels_str):
        if (i+1) % 100 == 0 or (i+1)*batch_size >= data_len:
            
            if (i+1)*batch_size >= data_len: # last batch
                data_count = data_len
                percent = 100
                ending = '\n'
            else:
                data_count = (i+1)*batch_size
                percent = data_count / data_len * 100
                ending = '\n'
            
            if not tune_parameter:
                print(f'Test Epoch: {epoch:3} [{data_count:5}/{data_len} ({percent:3.0f}%)]\tLoss: {loss.item():.3f}\tpred: {output_str[0]}, target: {labels_str[0]}', end=ending)
            else:
                print(f'Test Epoch: {epoch:3} [{data_count:5}/{data_len} ({percent:3.0f}%)]\tLoss: {loss.item():.3f}', end=ending)
    
    @staticmethod   
    def print_epoch_valid(epoch, data_len, test_loss, test_acc, duration):
        print(f' ||  Validation  Epoch: {epoch:3} {data_len:5}  Loss:[{test_loss:8.5f}]  ACC:[{test_acc*100:7.3f}]  --  time: ({duration:5.2f})')
        # print(f'Test set: LER: {test_ler:.3f}, PER: {test_per:.3f}, loss:{test_loss:.3f}', end='\n')
    @staticmethod   
    def print_epoch_test(snr, data_len, test_loss, test_acc, duration):
        print(f' ||  Test  SNR: {snr:3} {data_len:5}  Loss:[{test_loss:8.5f}]  ACC:[{test_acc*100:7.3f}]  --  time: ({duration:5.2f})')
        # print(f'Test set: LER: {test_ler:.3f}, PER: {test_per:.3f}, loss:{test_loss:.3f}', end='\n')
    
def main():
    GSC_CF = Trainer()
    GSC_CF.train_and_test()

if __name__ == '__main__':
    main()