"""
pipeline for building the training datasets
"""
import numpy as np
import h5py
from tqdm import tqdm
from pathlib import Path
import time
import torch
import sys

sys.path.append('./src')

from config.load_config import load_config
from parse_data.parse_trial_data import parse_trial_data
from dataset.data_process import (
    probR_sampling_for_choice,
    probR_threshold_for_choice,
    seqC_nan2num_norm,
    seqC_pattern_summary,
    reshape_shuffle_x_theta,
)

class training_dataset:
    """
    training dataset class
    """

    def __init__(self, config):

        self.config = config
        self._get_params()

    def _get_params(self):
        """
        get the parameters of the dataset
        """
        config = self.config
        self.num_probR_sample = self._get_num_probR_sample()

    def _get_num_probR_sample(self):

        Rchoice_method = self.config['dataset']['Rchoice_method']
        
        if Rchoice_method == 'probR_sampling':
            num_probR_sample = self.config['dataset']['num_probR_sample']
        else:
            num_probR_sample = 1

        return num_probR_sample


    def _process_x_R_part(self, probR_sub):
        """ 
        probR_sub: ndarray, shape (D,M,S,T, 1)
        output the R part of the input x of shape (D, M, S, T, C, 1)
        """
        R_method = self.config['dataset']['Rchoice_method']
        # if len(probR_sub.shape) <= 1:
        #     probR_sub = probR_sub.reshape(1, -1)

        if R_method == 'probR':
            R_part = np.expand_dims(probR_sub, axis=-2)
        elif R_method == 'probR_sampling':
            R_part = probR_sampling_for_choice(probR_sub, self.num_probR_sample)
        elif R_method == 'probR_threshold':
            R_part = probR_threshold_for_choice(probR_sub, self.config['dataset']['R_threshold'])
        else:
            raise ValueError(f'Invalid Rchoice_method: {R_method}')

        return R_part


    def _process_x_seqC_part(self, seqC):

        seqC_process_method = self.config['dataset']['seqC_process']
        if seqC_process_method == 'norm':
            nan2num = self.config['dataset']['nan2num']
            seqC = seqC_nan2num_norm(seqC, nan2num=nan2num)
        elif seqC_process_method == 'summary':
            summary_type = self.config['dataset']['summary_type']
            seqC = seqC_pattern_summary(seqC, summary_type=summary_type)
        else:
            raise ValueError(f'Invalid seqC_process: {seqC_process_method}')

        return seqC


    def data_process_pipeline(self, seqC, theta, probR, 
                              save_data_path=None):
        '''
        pipeline for processing the seqC, theta, probR for training

        Args:
            seqC : ndarray, shape (D,M,S,T, 15)
            theta: ndarray, shape (D,M,S,T, 5)
            probR: ndarray, shape (D,M,S,T, 1)

        1. do choice sampling for probR and process seqC, theta
            x_seqC: ndarray, shape (D,M,S,T,C, 15)
            theta_: ndarray, shape (D,M,S,T,C, 5)
        2. then reshape and shuffle for training
        
        Returns:
            x     (torch.tensor): shape (T*C, D*M*S, L_x)
            theta (torch.tensor): shape (T*C, L_theta)
        '''

        print(f'\n--- processing for x, theta with ---\ninputs: seqC.shape {seqC.shape}, theta.shape {theta.shape}, probR.shape {probR.shape}')
        # process seqC
        seqC   = seqC[:,:,:,:, np.newaxis, :]
        x_seqC = np.repeat(seqC, self.num_probR_sample, axis=4)
        x_seqC = self._process_x_seqC_part(x_seqC)
        #  x_seqC.shape = (D,M,S,T,C, 15)
        
        # process theta
        theta  = theta[:,:,:,:, np.newaxis, :] 
        theta_ = np.repeat(theta, self.num_probR_sample, axis=4)
        # theta_ = self._process_theta(theta_)
        #  theta_.shape = (D,M,S,T,C, 4)
        
        # process probR
        x_ch = self._process_x_R_part(probR)

        x = np.concatenate([x_seqC, x_ch], axis=-1)
        #  x.shape = (D,M,S,T,C, 16)
        
        # print(f'x.shape = {x.shape}, theta.shape = {theta.shape}')
        x, theta = reshape_shuffle_x_theta(x, theta_)
        
        if save_data_path != None:
            # save dataset
            f = h5py.File(save_data_path, 'w')
            f.create_dataset('x', data=x)
            f.create_dataset('theta', data=theta)
            f.close()
            print(f'training dataset saved to {save_data_path}')
        # else:
        #     print('you chose not to save the training data')
        
        return x, theta

    def x_seqC_data_process(self, seqC):
        '''
        pipeline for processing the seqC for training

        Args:
            seqC : ndarray, shape (D,M,S, 15)
            theta: ndarray, shape (T, 4)
            probR: ndarray, shape (D,M,S,T, 1)

        1. do choice sampling for probR and process seqC, theta
            x_seqC: ndarray, shape (D,M,S,T,C, 15)
            theta_: ndarray, shape (D,M,S,T,C, 5)
        2. then reshape and shuffle for training
        
        Returns:
            x     (torch.tensor): shape (T*C, D*M*S, L_x)
            theta (torch.tensor): shape (T*C, L_theta)
        '''

        print(f'\n--- processing for x, theta with ---\ninputs: seqC.shape {seqC.shape}, theta.shape {theta.shape}, probR.shape {probR.shape}')
        # process seqC
        seqC   = seqC[:,:,:,:, np.newaxis, :]
        x_seqC = np.repeat(seqC, self.num_probR_sample, axis=4)
        x_seqC = self._process_x_seqC_part(x_seqC)
        #  x_seqC.shape = (D,M,S,T,C, 15)
        
        # process theta
        theta  = theta[:,:,:,:, np.newaxis, :] 
        theta_ = np.repeat(theta, self.num_probR_sample, axis=4)
        # theta_ = self._process_theta(theta_)
        #  theta_.shape = (D,M,S,T,C, 4)
        
        # process probR
        x_ch = self._process_x_R_part(probR)

        x = np.concatenate([x_seqC, x_ch], axis=-1)
        #  x.shape = (D,M,S,T,C, 16)
        
        # print(f'x.shape = {x.shape}, theta.shape = {theta.shape}')
        x, theta = reshape_shuffle_x_theta(x, theta_)
        
        if save_data_path != None:
            # save dataset
            f = h5py.File(save_data_path, 'w')
            f.create_dataset('x', data=x)
            f.create_dataset('theta', data=theta)
            f.close()
            print(f'training dataset saved to {save_data_path}')
        # else:
        #     print('you chose not to save the training data')
        
        return x, theta

    
if __name__ == '__main__':
    # load and merge yaml files
    debug = False
    
    if debug: 
        f = h5py.File('../data/datasets/test_sim_data_seqC_prior_pR.h5', 'r')

        seqC = f['data_group']['seqCs'][:]
        theta = f['data_group']['theta'][:]
        probR = f['data_group']['probR'][:]
        
        seqC_pattern_summary(seqC, summary_type=1)
        seqC = seqC_nan2num_norm(seqC)
        choice = probR_sampling_for_choice(probR, num_probR_sample=10)
        choice = probR_threshold_for_choice(probR, threshold=0.5)
    
    test = True

    if test:
        config = load_config(
            config_simulator_path=Path('./src/config') / 'test' / 'test_simulator.yaml',
            config_dataset_path=Path('./src/config') / 'test' / 'test_dataset.yaml',
            config_train_path=Path('./src/config') / 'test' / 'test_train.yaml',
        )
    else:
        config = load_config(
            config_simulator_path=Path('./src/config') / 'simulator' / 'simulator_Ca_Pa_Ma.yaml',
            config_dataset_path=Path('./src/config') / 'dataset' / 'dataset_Sa0_Ra_suba0.yaml',
            config_train_path=Path('./src/config') / 'train' / 'train_Ta0.yaml',
        )
    print(config.keys())

    # # loading result from stored simulation file
    f = h5py.File(Path(config['data_dir']) / config['simulator']['save_name']+'run0.h5', 'r')
    seqC, theta, probR = f['data_group']['seqCs'][:], f['data_group']['theta'][:], f['data_group']['probR'][:]
    
    save_data_path = Path(config['data_dir']) / config['dataset']['save_name']+f'run{0}.h5'
    
    dataset = training_dataset(config)
    x, theta = dataset.data_process_pipeline(seqC, theta, probR, 
                                             save_data_path=save_data_path)
    # x, theta = dataset.subset_process_save_dataset(sim_data_dir)
    
    print('x.shape =', x.shape, 'theta.shape =', theta.shape)
    
    #! x     (torch.tensor): shape (T, C, D*M*S, L_x)
    #! theta (torch.tensor): shape (T, C, L_theta)