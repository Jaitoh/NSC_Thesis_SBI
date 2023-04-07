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


def reshape_shuffle_x_theta(x, theta):
    """do reshape for x and theta for network input

    Args:
        x     (np.array): shape (D,M,S,T,C, L_x)
        theta (np.array): shape (D,M,S,T,C, L_theta)
    
    Return:
        x     (torch.tensor): shape (T*C, D*M*S, L_x)
        theta (torch.tensor): shape (T*C, L_theta)
    """
    print(f'before reshape_shuffle_x_theta: \nx.shape={x.shape}, theta.shape={theta.shape}')
    D,M,S,T,C,L_x = x.shape
    # 0,1,2,3,4,5
    _,_,_,_,_,L_theta = theta.shape
    x = torch.from_numpy(x)
    theta = torch.from_numpy(theta)
    
    x_ = x.permute(0,3,4,5,1,2).reshape(D,T,C,L_x,M*S) # D,T,C,L_x,M*S
    x_ = x_.permute(0,1,2,4,3)  # D,T,C,M*S,L_x
                                # 0,1,2,3  ,4

    theta_ = theta.permute(0,3,4,5,1,2).reshape(D,T,C,L_theta,M*S) # D,T,C,L_theta,M*S
    theta_ = theta_.permute(0,1,2,4,3)  # D,T,C,M*S,L_theta
                                        # 0,1,2,3  ,4

    # randomize the order of the same duration D
    x_processed = torch.empty_like(x_)
    theta_processed = torch.empty_like(theta_)
    for d in range(D):
        for t in range(T): #TODO ! check dimension
            # for c in range(C):
            x_temp = x_[d,t,:,:,:]
            theta_temp = theta_[d,t,:,:,:]
            
            idx = torch.randperm(M*S)
            x_processed[d,t,:,:,:] = x_temp[:,idx,:] # M*S,L_x
            theta_processed[d,t,:,:,:] = theta_temp[:,idx,:] # M*S,L_theta

    # further reshape x_processed and theta_processed
    x_ = x_processed
    theta_ = theta_processed
    
    # to shape T,C,L_x,D,M*S
    x_ = x_.permute(1,2,4,0,3).reshape(T,C,L_x,D*M*S)# T,C,L_x,D*M*S
                                                     # 0,1,2  ,3
    # to shape L_x, D*M*S, T, C
    x_ = x_.permute(2, 3, 0, 1).reshape(L_x, D*M*S, T*C) # L_x, D*M*S, T*C
                                                        # 0  ,1     ,2
    x_ = x_.permute(2,1,0) # T*C, D*M*S, L_x
    # to shape T, C, D*M*S, L_x
    # x_ = x_.permute(0,1,3,2)

    # to shape T,C,L_theta,D,M*S
    theta_ = theta_.permute(1,2,4,0,3).reshape(T,C,L_theta,D*M*S) # T,C,L_theta,D*M*S
                                                                  # 0,1,2      ,3
    # to shape L_theta, D*M*S, T, C
    theta_ = theta_.permute(2, 3, 0, 1).reshape(L_theta, D*M*S, T*C) # L_theta, D*M*S, T*C
                                                                    # 0      ,1     ,2 
    theta_ = theta_.permute(2,1,0) # T*C, D*M*S, L_theta
    # to shape T, C, D*M*S, L_theta
    # theta_ = theta_.permute(0,1,3,2)

    # output shape: T*C, 0, L_theta
    theta_ = theta_[:,0,:]
    print('reshaped and shuffled: \nx.shape', x_.shape, 'theta.shape', theta_.shape)
    
    return x_, theta_

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
    MS      = np.apply_along_axis(lambda x: np.unique(np.abs(x[(~np.isnan(x))&(x!=0)])), axis=-1, arr=seqC)
    MS      = np.reshape(MS, (*seqC.shape[:-1], )) # MS = MS[:,:,:,:,:,-1]
    
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
    probR = np.reshape(probR, (*probR.shape[:-1], ))
    
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


# def process_x_seqC_part(seqC, config):

#     # input seqC is a 2D array with shape (num_seqC, seqC_len)

#     if len(seqC.shape) == 1:
#         seqC = seqC.reshape(1, -1)

#     seqC_process_method = config['dataset']['seqC_process']
#     if seqC_process_method == 'norm':
#         nan2num = config['dataset']['nan2num']
#         seqC = seqC_nan2num_norm(seqC, nan2num=nan2num)
#     elif seqC_process_method == 'summary':
#         summary_type = config['dataset']['summary_type']
#         seqC = seqC_pattern_summary(seqC, summary_type=summary_type)
#     else:
#         raise ValueError(f'Invalid seqC_process: {seqC_process_method}')

#     return seqC

def process_x_seqC_part(
    seqC, 
    seqC_process_method,
    nan2num,
    summary_type,
):

    # input seqC is a 2D array with shape (num_seqC, seqC_len)

    if len(seqC.shape) == 1:
        seqC = seqC.reshape(1, -1)

    # seqC_process_method = config['dataset']['seqC_process']
    if seqC_process_method == 'norm':
        # nan2num = config['dataset']['nan2num']
        seqC = seqC_nan2num_norm(seqC, nan2num=nan2num)
    elif seqC_process_method == 'summary':
        # summary_type = config['dataset']['summary_type']
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

        # self.num_seqC_sample = config['seqC']['sample']
        # self.num_prior_sample = config['prior']['num_prior_sample']

        # self.MS_list = config['seqC']['MS_list']
        # self.prior_min = config['prior']['prior_min']

        self.num_probR_sample = self._get_num_probR_sample()

    def _get_num_probR_sample(self):

        Rchoice_method = self.config['dataset']['Rchoice_method']
        if Rchoice_method == 'probR_sampling':
            num_probR_sample = self.config['dataset']['num_probR_sample']
        else:
            num_probR_sample = 1

        return num_probR_sample


    def _process_x_R_part(self, probR_sub):
        """ output the R part of the input x of shape (D, M, S, T, C, 1)"""
        R_method = self.config['dataset']['Rchoice_method']
        if len(probR_sub.shape) <= 1:
            probR_sub = probR_sub.reshape(1, -1)

        if R_method == 'probR':
            R_part = np.expand_dims(probR_sub, axis=-1)
        elif R_method == 'probR_sampling':
            R_part = probR_sampling_for_choice(probR_sub, self.num_probR_sample)
        elif R_method == 'probR_threshold':
            R_part = probR_threshold_for_choice(probR_sub, self.config['dataset']['R_threshold'])
        else:
            raise ValueError(f'Invalid Rchoice_method: {R_method}')

        return R_part


    def _process_x_seqC_part(self, seqC):

        if len(seqC.shape) == 1:
            seqC = seqC.reshape(1, -1)

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


    def _process_theta(self, theta):

        if len(theta.shape) == 1:
            theta = theta.reshape(1, -1)

        if self.config['dataset']['remove_sigma2i']:
            theta = np.delete(theta, 2, axis=-1)  # bsssxxx remove sigma2i (the 3rd column)

        return theta


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

        print(f'---\nstart processing for x, theta \nwith inputs: seqC.shape {seqC.shape}, theta.shape {theta.shape}, probR.shape {probR.shape}')
        # process seqC
        seqC   = seqC[:,:,:,:, np.newaxis, :]
        x_seqC = np.repeat(seqC, self.num_probR_sample, axis=4)
        x_seqC = self._process_x_seqC_part(x_seqC)
        # TODO x_seqC.shape = (D,M,S,T,C, 15)
        
        # process theta
        theta  = theta[:,:,:,:, np.newaxis, :] 
        theta_ = np.repeat(theta, self.num_probR_sample, axis=4)
        # theta_ = self._process_theta(theta_)
        # TODO theta_.shape = (D,M,S,T,C, 4)
        
        # process probR
        x_ch = self._process_x_R_part(probR)

        x = np.concatenate([x_seqC, x_ch], axis=-1)
        # TODO x.shape = (D,M,S,T,C, 16)
        
        # print(f'x.shape = {x.shape}, theta.shape = {theta.shape}')
        x, theta = reshape_shuffle_x_theta(x, theta_)
        
        if save_data_path != None:
            # save dataset
            f = h5py.File(save_data_path, 'w')
            f.create_dataset('x', data=x)
            f.create_dataset('theta', data=theta)
            f.close()
            print(f'training dataset saved to {save_data_path}')
        else:
            print(f'you chose not to save the training data')
        
        return x, theta

    # def get_subset_data(self, seqC, theta, probR):
    #     """get the subset of the training dataset based on the config

    #     Args:
    #         seqC  (np.ndarray): shape (D, M, S, T, 15)
    #         theta (np.ndarray): shape (D, M, S, T, 5)
    #         probR (np.ndarray): shape (D, M, S, T, 1)

    #     Returns:
    #         subset of seqC, theta, probR
    #     """
    #     # 1. take part of dur, MS
    #     train_data_dur_list = config['dataset']['train_data_dur_list']
        train_data_MS_list = config['dataset']['train_data_MS_list']
        
        # [get corresponding idx] decode train_data_dur_list/MS_list to list_idx -> find corresponding idx number in the list
    #     dur_min, dur_max, dur_step = config['seqC']['dur_min'], config['seqC']['dur_max'], config['seqC']['dur_step']
    #     dur_list = list(np.arange(dur_min, dur_max+1, dur_step))
    #     train_data_dur_list_idx = [dur_list.index(dur) for dur in train_data_dur_list] # e.g. [4,5]
        train_data_MS_list_idx = [self.MS_list.index(MS) for MS in train_data_MS_list] # e.g. [0,2]
        
    #     seqC_sub  = seqC[train_data_dur_list_idx, :, :, :, :]
        seqC_sub  = seqC_sub[:, train_data_MS_list_idx, :, :, :]
    #     theta_sub = theta[train_data_dur_list_idx, :, :, :, :]
        theta_sub = theta_sub[:, train_data_MS_list_idx, :, :, :]
    #     probR_sub = probR[train_data_dur_list_idx, :, :, :, :]
        probR_sub = probR_sub[:, train_data_MS_list_idx, :, :, :]
        
    #     # 2. take part of seqC, theta content
    #     subset_seqC = config['dataset']['subset_seqC']
    #     subset_theta = config['dataset']['subset_theta']
        
    #     S, T = seqC_sub.shape[2], seqC_sub.shape[3]
    #     if isinstance(subset_seqC, list):
    #         subset_S_list = subset_seqC
    #     else:
    #         subset_S_list = list(np.arange(int(subset_seqC*S)))
    #         assert len(subset_S_list) != 0, 'subset_seqC lead to a empty list'

    #     if isinstance(subset_theta, list):
    #         subset_T_list = subset_theta
    #     else:
    #         subset_T_list = list(np.arange(int(subset_theta*T)))
    #         assert len(subset_T_list) != 0, 'subset_theta lead to a empty list'

    #     seqC_sub  = seqC_sub [:, :, subset_S_list, :, :]
    #     theta_sub = theta_sub[:, :, subset_S_list, :, :]
    #     probR_sub = probR_sub[:, :, subset_S_list, :, :]
    #     seqC_sub  = seqC_sub [:, :, :, subset_T_list, :]
    #     theta_sub = theta_sub[:, :, :, subset_T_list, :]
    #     probR_sub = probR_sub[:, :, :, subset_T_list, :]
        
    #     return seqC_sub, theta_sub, probR_sub

    # def subset_process_save_dataset(self, data_dir, save_train_data=False):
    #     """
    #     Generate dataset for training
    #     Args:
    #         data_dir: data_dir saved the sim_data h5 file

    #     Returns:
    #         x     (torch.tensor): shape (T*C, D*M*S, L_x)
    #         theta (torch.tensor): shape (T*C, L_theta)
    #     """
    #     # load sim_data h5 file
    #     f = h5py.File(Path(data_dir) / self.config['simulator']['save_name'], 'r')
        
    #     print('loading and processing subset data...')
    #     start_time = time.time()
    #     seqC, theta, probR = f['data_group']['seqCs'][:], f['data_group']['theta'][:], f['data_group']['probR'][:]
    #     f.close()
        
    #     seqC_sub, theta_sub, probR_sub = self.get_subset_data(seqC, theta, probR)
    #     x, theta = self.data_process_pipeline(seqC_sub, theta_sub, probR_sub)
    #     print(f'finished loading and processing of subset dataset, time used: {time.time() - start_time:.2f}s')
        
    #     if save_train_data:
    #         # save dataset
    #         save_data_path = Path(data_dir) / self.config['dataset']['save_name']
    #         f = h5py.File(save_data_path, 'w')
    #         f.create_dataset('x', data=x)
    #         f.create_dataset('theta', data=theta)
    #         f.close()
    #         print(f'training dataset saved to {save_data_path}')

    #     return x, theta


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