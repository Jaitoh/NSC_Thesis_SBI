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

# from dataset.seqC_pR_process import seqC_pattern_summary, probR_sampling_for_choice, probR_threshold_for_choice, \
#     seqC_nan2num_norm
from config.load_config import load_config


def seqC_nan2num_norm(seqC, nan2num=-1):
    """ fill the nan of the seqC with nan2num and normalize to (0, 1)
    """
    seqC = np.nan_to_num(seqC, nan=nan2num)
    # normalize the seqC from (nan2num, 1) to (0, 1)
    seqC = (seqC - nan2num) / (1 - nan2num)

    return seqC


def seqC_pattern_summary(seqC, summary_type=1, dur_max=15):

    """ extract the input sequence pattern summary from the input seqC

        can either input a array of shape (D,M,S,T,C, 15)
        or a dictionary of pulse sequences contain all the information listed below for the further computation
        
        Args:
            seqC (np.array): input sequence of shape (D,M,S,T, 15)  !should be 2 dimensional
                e.g.  np.array([[0, 0.4, -0.4, 0, 0.4, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                [0, 0.4, -0.4, 0, 0.4, 0.4, -0.4, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])

            summary_type:   (default: 1)
                0: with separate left and right (same/oppo/new)
                1: combine left and right (same/oppo/new)

        Return:
            summary_type 0:
            x0 (np.array): input pattern summary of shape (D,M,S,T,C, 11)
                column 1: MS
                column 2: dur
                column 3: nLeft
                column 4: nRight
                column 5: nPulse
                column 6: hist_nLsame
                column 7: hist_nLoppo
                column 8: hist_nLelse
                column 9: hist_nRsame
                column 10: hist_nRoppo
                column 11: hist_nRelse

            summary_type 1:
            x1 (np.array): input pattern summary of shape (D,M,S,T,C, 8)
                column 1: MS
                column 2: dur
                column 3: nLeft
                column 4: nRight
                column 5: nPulse
                column 6: hist_nSame
                column 7: hist_nOppo
                column 8: hist_nElse
                
    """
    
    # get the MS of each trial
    MS      = np.apply_along_axis(lambda x: np.unique(np.abs(x[(~np.isnan(x))&(x!=0)])), axis=-1, arr=seqC)[:,:,:,:,:,-1]
    _dur    = np.apply_along_axis(lambda x: np.sum(~np.isnan(x)), axis=-1, arr=seqC)
    _nLeft  = np.apply_along_axis(lambda x: np.sum(x<0), axis=-1, arr=seqC)
    _nRight = np.apply_along_axis(lambda x: np.sum(x>0), axis=-1, arr=seqC)
    _nPulse = _dur - _nLeft - _nRight

    # summary of effect stimulus
    dur     = (_dur-1)/(dur_max-1)
    nLeft   = _nLeft/(dur_max-1)
    nRight  = _nRight/(dur_max-1)
    nPause  = (_dur-1-_nLeft-_nRight)/(dur_max-1)

    # extract internal pattern summary
    hist_nSame  = np.apply_along_axis(lambda x: np.sum(x*np.append(0, x[0:-1])>0), axis=-1, arr=seqC)/(_dur-1)
    hist_nLsame = np.apply_along_axis(lambda x: np.sum((x*np.append(0, x[0:-1])>0) & (x<0)), axis=-1, arr=seqC)/(_dur-1)
    hist_nRsame = np.apply_along_axis(lambda x: np.sum((x*np.append(0, x[0:-1])>0) & (x>0)), axis=-1, arr=seqC)/(_dur-1)

    hist_nOppo  = np.apply_along_axis(lambda x: np.sum(x*np.append(0, x[0:-1])<0), axis=-1, arr=seqC)/(_dur-1)
    hist_nLoppo = np.apply_along_axis(lambda x: np.sum((x*np.append(0, x[0:-1])<0) & (x<0)), axis=-1, arr=seqC)/(_dur-1)
    hist_nRoppo = np.apply_along_axis(lambda x: np.sum((x*np.append(0, x[0:-1])<0) & (x>0)), axis=-1, arr=seqC)/(_dur-1)

    hist_nElse  = np.apply_along_axis(lambda x: np.sum( (x*np.append(0, x[0:-1])==0) & (x!=0) ), axis=-1, arr=seqC)/(_dur-1)
    hist_nLelse = np.apply_along_axis(lambda x: np.sum( (x*np.append(0, x[0:-1])==0) & (x<0) ), axis=-1, arr=seqC)/(_dur-1)
    hist_nRelse = np.apply_along_axis(lambda x: np.sum( (x*np.append(0, x[0:-1])==0) & (x>0) ), axis=-1, arr=seqC)/(_dur-1)

    # add one more dimension for concatenation
    MS          = np.expand_dims(MS, axis=-1)
    dur         = np.expand_dims(dur, axis=-1)
    nLeft       = np.expand_dims(nLeft, axis=-1)
    nRight      = np.expand_dims(nRight, axis=-1)
    nPause      = np.expand_dims(nPause, axis=-1)
    hist_nLsame = np.expand_dims(hist_nLsame, axis=-1)
    hist_nLoppo = np.expand_dims(hist_nLoppo, axis=-1)
    hist_nLelse = np.expand_dims(hist_nLelse, axis=-1)
    hist_nRsame = np.expand_dims(hist_nRsame, axis=-1)
    hist_nRoppo = np.expand_dims(hist_nRoppo, axis=-1)
    hist_nRelse = np.expand_dims(hist_nRelse, axis=-1)
    hist_nSame  = np.expand_dims(hist_nSame, axis=-1)
    hist_nOppo  = np.expand_dims(hist_nOppo, axis=-1)
    hist_nElse  = np.expand_dims(hist_nElse, axis=-1)

    # concatenate the summary along the 5th dimension
    x0 = np.concatenate((MS, dur, nLeft, nRight, nPause, hist_nLsame, hist_nLoppo, hist_nLelse, hist_nRsame, hist_nRoppo, hist_nRelse), axis=-1)
    x1 = np.concatenate((MS, dur, nLeft, nRight, nPause, hist_nSame, hist_nOppo, hist_nElse), axis=-1)

    if summary_type == 0:
        return x0
    else:  # default output
        return x1


def probR_sampling_for_choice(probR, num_probR_sample=10):
    """ sample the probability of right choice from the input probR

        Args:
            probR (np.array): input probability of right choice of shape (D,M,S,T, 1)
            num_probR_sample (int): number of samples for each input probability of right choice

        Return:
            probR_sample (np.array): sampled probability of right choice of shape (D,M,S,T, num_probR_sample(C), 1)
    """
    if not isinstance(probR, np.ndarray):
        probR = np.array(probR).reshape(-1, 1)
    probR = np.squeeze(probR)
    
    choice = np.empty((*probR.shape, num_probR_sample))
    for D in range(probR.shape[0]):
        for M in range(probR.shape[1]):
            for S in range(probR.shape[2]):
                for T in range(probR.shape[3]):
                    prob = probR[D, M, S, T]
                    # choice[D, M, S, :] = np.random.choice([0, 1], size=num_probR_sample, p=[1 - prob, prob])
                    cs = np.random.binomial(1, prob, size=num_probR_sample)
                    choice[D, M, S, T, :] = cs
    choice = choice[:, :, :, :, :, np.newaxis]
    # TODO choice.shape = (D,M,S,T, num_probR_sample)
    return choice


def probR_threshold_for_choice(probR, threshold=0.5):
    """ get right choice from the probR, when probR > threshold, choose right(1) else Left(0)

        Args:
            probR (np.array): input probability of right choice of shape (D,M,S,T, 1)
            threshold (float): threshold for right choice

        Return:
            choice (np.array): sampled probability of right choice of shape (D,M,S,T,C, 1)
    """

    if not isinstance(probR, np.ndarray):
        probR = np.array(probR).reshape(-1, 1)
    choice = np.where(probR > threshold, 1, 0)
    choice = choice[:, :, :, :, :, np.newaxis]
    return choice


def process_x_seqC_part(seqC, config):

    # input seqC is a 2D array with shape (num_seqC, seqC_len)

    if len(seqC.shape) == 1:
        seqC = seqC.reshape(1, -1)

    seqC_process_method = config['train_data']['seqC_process']
    if seqC_process_method == 'norm':
        nan2num = config['train_data']['nan2num']
        seqC = seqC_nan2num_norm(seqC, nan2num=nan2num)
    elif seqC_process_method == 'summary':
        summary_type = config['train_data']['summary_type']
        seqC = seqC_pattern_summary(seqC, summary_type=summary_type)
    else:
        raise ValueError(f'Invalid seqC_process: {seqC_process_method}')

    return seqC

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

        self.num_seqC_sample = config['seqC']['sample']
        self.num_prior_sample = config['prior']['num_prior_sample']

        self.MS_list = config['seqC']['MS_list']
        self.prior_min = config['prior']['prior_min']

        self.num_probR_sample = self._get_num_probR_sample()

    def get_subset_data(self, seqC, theta, probR):
        """get the subset of the training dataset based on the config

        Args:
            seqC  (np.ndarray): shape (D, M, S, T, 15)
            theta (np.ndarray): shape (D, M, S, T, 5)
            probR (np.ndarray): shape (D, M, S, T, 1)

        Returns:
            subset of seqC, theta, probR
        """
        # 1. take part of dur, MS
        train_data_dur_list = config['train_data']['train_data_dur_list']
        train_data_MS_list = config['train_data']['train_data_MS_list']
        
        # [get corresponding idx] decode train_data_dur_list/MS_list to list_idx -> find corresponding idx number in the list
        dur_min, dur_max, dur_step = config['seqC']['dur_min'], config['seqC']['dur_max'], config['seqC']['dur_step']
        dur_list = list(np.arange(dur_min, dur_max+1, dur_step))
        train_data_dur_list_idx = [dur_list.index(dur) for dur in train_data_dur_list] # e.g. [4,5]
        train_data_MS_list_idx = [self.MS_list.index(MS) for MS in train_data_MS_list] # e.g. [0,2]
        
        seqC_sub  = seqC[train_data_dur_list_idx, :, :, :, :]
        seqC_sub  = seqC_sub[:, train_data_MS_list_idx, :, :, :]
        theta_sub = theta[train_data_dur_list_idx, :, :, :, :]
        theta_sub = theta_sub[:, train_data_MS_list_idx, :, :, :]
        probR_sub = probR[train_data_dur_list_idx, :, :, :, :]
        probR_sub = probR_sub[:, train_data_MS_list_idx, :, :, :]
        
        # 2. take part of seqC, theta content
        subset_seqC = config['train_data']['subset_seqC']
        subset_theta = config['train_data']['subset_theta']
        
        S, T = seqC_sub.shape[2], seqC_sub.shape[3]
        if isinstance(subset_seqC, list):
            subset_S_list = subset_seqC
        else:
            subset_S_list = list(np.arange(int(subset_seqC*S)))
            assert len(subset_S_list) != 0, 'subset_seqC lead to a empty list'

        if isinstance(subset_theta, list):
            subset_T_list = subset_theta
        else:
            subset_T_list = list(np.arange(int(subset_theta*T)))
            assert len(subset_T_list) != 0, 'subset_theta lead to a empty list'

        seqC_sub  = seqC_sub [:, :, subset_S_list, :, :]
        theta_sub = theta_sub[:, :, subset_S_list, :, :]
        probR_sub = probR_sub[:, :, subset_S_list, :, :]
        seqC_sub  = seqC_sub [:, :, :, subset_T_list, :]
        theta_sub = theta_sub[:, :, :, subset_T_list, :]
        probR_sub = probR_sub[:, :, :, subset_T_list, :]
        
        return seqC_sub, theta_sub, probR_sub
    
    
    def _get_num_probR_sample(self):

        Rchoice_method = self.config['train_data']['Rchoice_method']
        if Rchoice_method == 'probR_sampling':
            num_probR_sample = self.config['train_data']['num_probR_sample']
        else:
            num_probR_sample = 1

        return num_probR_sample


    def _process_x_R_part(self, probR_sub):
        """ output the R part of the input x of shape (D, M, S, T, C, 1)"""
        R_method = self.config['train_data']['Rchoice_method']
        if len(probR_sub.shape) <= 1:
            probR_sub = probR_sub.reshape(1, -1)

        if R_method == 'probR':
            R_part = np.expand_dims(probR_sub, axis=-1)
        elif R_method == 'probR_sampling':
            R_part = probR_sampling_for_choice(probR_sub, self.num_probR_sample)
        elif R_method == 'probR_threshold':
            R_part = probR_threshold_for_choice(probR_sub, self.config['train_data']['R_threshold'])
        else:
            raise ValueError(f'Invalid Rchoice_method: {R_method}')

        return R_part


    def _process_x_seqC_part(self, seqC):

        if len(seqC.shape) == 1:
            seqC = seqC.reshape(1, -1)

        seqC_process_method = self.config['train_data']['seqC_process']
        if seqC_process_method == 'norm':
            nan2num = self.config['train_data']['nan2num']
            seqC = seqC_nan2num_norm(seqC, nan2num=nan2num)
        elif seqC_process_method == 'summary':
            summary_type = self.config['train_data']['summary_type']
            seqC = seqC_pattern_summary(seqC, summary_type=summary_type)
        else:
            raise ValueError(f'Invalid seqC_process: {seqC_process_method}')

        return seqC


    def _process_theta(self, theta):

        if len(theta.shape) == 1:
            theta = theta.reshape(1, -1)

        if self.config['train_data']['remove_sigma2i']:
            theta = np.delete(theta, 2, axis=-1)  # bsssxxx remove sigma2i (the 3rd column)

        return theta


    def data_process_pipeline(self, seqC, theta, probR):
        '''
        pipeline for processing the seqC, theta, probR for training

        Args:
            seqC : ndarray, shape (D,M,S,T, 15)
            theta: ndarray, shape (D,M,S,T, 5)
            probR: ndarray, shape (D,M,S,T, 1)

        Returns:
            x_seqC: ndarray, shape (D,M,S,T,C, 15)
            theta_: ndarray, shape (D,M,S,T,C, 5)
            x_R   : ndarray, shape (D,M,S,T,C, 1)
        '''

        # process seqC
        seqC   = seqC[:,:,:,:, np.newaxis, :]
        x_seqC = np.repeat(seqC, self.num_probR_sample, axis=4)
        x_seqC = self._process_x_seqC_part(x_seqC)
        # TODO x_seqC.shape = (D,M,S,T,C, 15)
        
        # process theta
        theta  = theta[:,:,:,:, np.newaxis, :] 
        theta_ = np.repeat(theta, self.num_probR_sample, axis=4)
        theta_ = self._process_theta(theta_)
        # TODO theta_.shape = (D,M,S,T,C, 4)
        
        # process probR
        x_ch = self._process_x_R_part(probR)

        x = np.concatenate([x_seqC, x_ch], axis=-1)
        # TODO x.shape = (D,M,S,T,C, 16)
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(theta_, dtype=torch.float32)


    def subset_process_sava_dataset(self, data_dir):
        """
        Generate dataset for training
        Args:
            data_dir: data_dir saved the sim_data h5 file

        Returns:

        """
        # load sim_data h5 file
        f = h5py.File(Path(data_dir) / self.config['simulator']['save_name'], 'r')
        
        print('loading and processing subset data...')
        start_time = time.time()
        seqC, theta, probR = f['data_group']['seqCs'][:], f['data_group']['theta'][:], f['data_group']['probR'][:]
        f.close()
        
        seqC_sub, theta_sub, probR_sub = self.get_subset_data(seqC, theta, probR)
        x, theta = self.data_process_pipeline(seqC_sub, theta_sub, probR_sub)
        print(f'finished loading and processing of subset dataset, time used: {time.time() - start_time:.2f}s')
        
        # save dataset
        save_path = Path(data_dir) / self.config['train_data']['save_name']
        f = h5py.File(save_path, 'w')
        f.create_dataset('x', data=x)
        f.create_dataset('theta', data=theta)
        f.close()
        print(f'training dataset saved to {save_path}')

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

    sim_data_dir = config['data_dir']
    dataset = training_dataset(config)
    
    f = h5py.File(Path(sim_data_dir) / config['simulator']['save_name'], 'r')
    seqC, theta, probR = f['data_group']['seqCs'][:], f['data_group']['theta'][:], f['data_group']['probR'][:]
    x, theta = dataset.data_process_pipeline(seqC, theta, probR)
    x, theta = dataset.subset_process_sava_dataset(sim_data_dir)
