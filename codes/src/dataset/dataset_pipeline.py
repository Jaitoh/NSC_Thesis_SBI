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

from dataset.seqC_pR_process import seqC_pattern_summary, probR_sampling_for_choice, probR_threshold_for_choice, \
    seqC_nan2num_norm
from config.load_config import load_config

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

        self.index_chosen_in1dur = self._get_subset_idx()
        print(f'---\nprepare training dataset\noriginal data size (single dur) {self.single_dur_data_len}, taken a subset size (single dur) of {len(self.index_chosen_in1dur)}')

    def _get_params(self):
        """
        get the parameters of the dataset
        """
        config = self.config

        self.num_seqC_sample = config['seqC']['sample']
        self.num_prior_sample = config['prior']['num_prior_sample']

        self.MS_list = config['seqC']['MS_list']
        self.prior_min = config['prior']['prior_min']

        self.single_dur_data_len = len(self.MS_list) * self.num_seqC_sample * self.num_prior_sample
        self.single_dur_seqC_shape = [self.single_dur_data_len, config['seqC']['dur_max']]
        self.single_dur_theta_shape = [self.single_dur_data_len, len(self.prior_min)]

        self.num_probR_sample = self._get_num_probR_sample()

    def __get_MS_seqC_theta_pointers(self):

        MS_pointers_in1dur = list(range(0, self.single_dur_data_len, self.single_dur_data_len // len(self.MS_list)))

        single_MS_len = self.single_dur_data_len // len(self.MS_list)
        seqC_pointers_in1MS = list(range(0, single_MS_len, self.num_prior_sample))

        theta_pointers_in1seqC = list(range(0, self.num_prior_sample))

        return MS_pointers_in1dur, seqC_pointers_in1MS, theta_pointers_in1seqC

    def __get_subset_of_MS_seqC_pointers(self, MS_pointers_in1dur, seqC_pointers_in1MS, theta_pointers_in1seqC,
                                         is_subset_seqC_list, is_subset_theta_list):

        MS_pointers_subset = [MS_pointers_in1dur[self.MS_list.index(MS)] for MS in self.train_data_MS_list]

        if is_subset_seqC_list:
            seqC_pointers_subset = [seqC_pointers_in1MS[idx] for idx in self.subset_seqC]
        else:
            num_seqC_pointers_in1MS = len(seqC_pointers_in1MS)
            seqC_pointers_subset = seqC_pointers_in1MS[0:int(num_seqC_pointers_in1MS * self.subset_seqC)]

        if is_subset_theta_list:
            theta_pointers_subset = [theta_pointers_in1seqC[idx] for idx in self.subset_theta]
        else:
            num_theta_pointers_in1seqC = len(theta_pointers_in1seqC)
            theta_pointers_subset = theta_pointers_in1seqC[0:int(num_theta_pointers_in1seqC * self.subset_theta)]

        return np.array(MS_pointers_subset), np.array(seqC_pointers_subset), np.array(theta_pointers_subset)

    def _get_subset_idx(self):
        """
        get the subset_idx of the training dataset based on the config
        """

        config = self.config
        self.train_data_dur_list = config['train_data']['train_data_dur_list']
        self.train_data_MS_list = config['train_data']['train_data_MS_list']
        self.subset_seqC = config['train_data']['subset_seqC']
        self.subset_theta = config['train_data']['subset_theta']

        is_subset_seqC_list = True if isinstance(self.subset_seqC, list) else False
        is_subset_theta_list = True if isinstance(self.subset_theta, list) else False

        MS_pointers_in1dur, seqC_pointers_in1MS, theta_pointers_in1seqC = self.__get_MS_seqC_theta_pointers()
        MS_pointers_subset_in1dur, seqC_pointers_subset_in1MS, theta_pointers_subset_in1seqC = self.__get_subset_of_MS_seqC_pointers(
            MS_pointers_in1dur, seqC_pointers_in1MS, theta_pointers_in1seqC,
            is_subset_seqC_list, is_subset_theta_list)

        # add MS pointers to seqC pointers
        seqC_pointers_subset_in1dur = np.array(
            [seqC_pointers_subset_in1MS + MS_pointer for MS_pointer in MS_pointers_subset_in1dur]).reshape(-1)
        # add seqC pointers to theta pointers
        theta_pointers_subset_in1dur = np.array(
            [theta_pointers_subset_in1seqC + seqC_pointer for seqC_pointer in seqC_pointers_subset_in1dur]).reshape(-1)

        index_chosen_in1dur = theta_pointers_subset_in1dur

        return index_chosen_in1dur

    def _get_num_probR_sample(self):

        Rchoice_method = self.config['train_data']['Rchoice_method']
        if Rchoice_method == 'probR_sampling':
            num_probR_sample = self.config['train_data']['num_probR_sample']
        else:
            num_probR_sample = 1

        return num_probR_sample

    def _get_subset_data(self, dur):
        seqC_sub = self.seqC_group[f'seqC_dur{dur}'][self.index_chosen_in1dur]
        theta_sub = self.theta_group[f'theta_dur{dur}'][self.index_chosen_in1dur]
        probR_sub = self.probR_group[f'probR_dur{dur}'][self.index_chosen_in1dur]
        return seqC_sub, theta_sub, probR_sub

    def _process_x_R_part(self, probR_sub):

        R_method = self.config['train_data']['Rchoice_method']
        if len(probR_sub.shape) <= 1:
            probR_sub = probR_sub.reshape(1, -1)

        if R_method == 'probR':
            R_part = probR_sub
        elif R_method == 'probR_sampling':
            R_part = probR_sampling_for_choice(probR_sub, self.num_probR_sample)
        elif R_method == 'probR_threshold':
            R_part = probR_threshold_for_choice(probR_sub, self.config['train_data']['R_threshold'])
        else:
            raise ValueError(f'Invalid Rchoice_method: {R_method}')

        return R_part

    def _process_x_seqC_part(self, seqC):

        # input seqC is a 2D array with shape (num_seqC, seqC_len)

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
            theta = np.delete(theta, 2, axis=1)  # bsssxxx remove sigma2i (the 3rd column)

        return theta

    def data_process_pipeline(self, seqC, theta, probR):
        '''
        pipeline for processing the seqC, theta, probR for training

        Args:
            seqC: ndarray, shape (num_seqC, seqC_len)
            theta: ndarray, shape (num_theta, theta_len)
            probR:  ndarray, shape (num_probR, probR_len)

        Returns:

        '''

        # process seqC
        x_seqC = np.repeat(seqC, self.num_probR_sample, axis=0)
        x_seqC = self._process_x_seqC_part(x_seqC)
        # process theta
        theta_ = np.repeat(theta, self.num_probR_sample, axis=0)
        theta_ = self._process_theta(theta_)
        # process probR
        x_R = self._process_x_R_part(probR)

        return x_seqC, theta_, x_R

    def generate_save_dataset(self, data_dir):
        """
        Generate dataset for training
        Args:
            data_dir: data_dir saved the sim_data h5 file

        Returns:

        """
        # define the shape of the dataset
        dataset_size = len(self.index_chosen_in1dur) * len(self.train_data_dur_list) * self.num_probR_sample
        if self.config['train_data']['seqC_process'] == 'summary':
            if self.config['train_data']['summary_type'] == 0:
                x_seqC_shape = [dataset_size, 11]
            if self.config['train_data']['summary_type'] == 1:
                x_seqC_shape = [dataset_size, 8]
        else:
            x_seqC_shape = [dataset_size, self.config['seqC']['dur_max']]
        theta_shape = [dataset_size, len(self.prior_min)-1] # -1 for remove sigma2i
        x_R_shape = [dataset_size, 1]

        x_seqC, theta, x_R = np.empty(x_seqC_shape), np.empty(theta_shape), np.empty(x_R_shape)

        # load sim_data h5 file
        f = h5py.File(Path(data_dir) / self.config['simulator']['save_name'], 'r')
        self.seqC_group, self.theta_group, self.probR_group = f['seqC_group'], f['theta_group'], f['probR_group']

        print('loading and processing subset data...')
        start_time = time.time()
        for i, dur in tqdm(enumerate(self.train_data_dur_list), total=len(self.train_data_dur_list)):
            seqC_sub, theta_sub, probR_sub = self._get_subset_data(dur)

            x_seqC_part, theta_part, x_R_part = self.data_process_pipeline(seqC_sub, theta_sub, probR_sub)

            part_len = len(self.index_chosen_in1dur) * self.num_probR_sample
            x_seqC[i * part_len:(i + 1) * part_len, :] = x_seqC_part
            theta[i * part_len:(i + 1) * part_len, :] = theta_part
            x_R[i * part_len:(i + 1) * part_len, :] = x_R_part

        f.close()
        print(f'finished loading and processing of subset dataset, time used: {time.time() - start_time:.2f}s')

        # process seqC
        # x_seqC = self._process_x_seqC_part(x_seqC)
        x = np.concatenate([x_seqC, x_R], axis=1)

        # process theta
        # theta_ = self._process_theta(theta)

        # save dataset
        save_path = Path(data_dir) / self.config['train_data']['save_name']
        f = h5py.File(save_path, 'w')
        f.create_dataset('x', data=x)
        f.create_dataset('theta', data=theta)
        f.close()
        print(f'training dataset saved to {save_path}')

        return torch.tensor(x, dtype=torch.float32), torch.tensor(theta, dtype=torch.float32)


if __name__ == '__main__':
    # load and merge yaml files
    test = True

    if test:
        config = load_config(
            config_simulator_path=Path('./src/config') / 'test_simulator.yaml',
            config_dataset_path=Path('./src/config') / 'test_dataset.yaml',
            config_train_path=Path('./src/config') / 'test_train.yaml',
        )
    else:
        config = load_config(
            config_simulator_path=Path('./src/config') / 'simulator_Ca_Pa_Ma.yaml',
            config_dataset_path=Path('./src/config') / 'dataset_Sa0_Ra_suba0.yaml',
            config_train_path=Path('./src/config') / 'train_Ta0.yaml',
        )
    print(config.keys())

    sim_data_dir = config['data_dir']
    dataset = training_dataset(config)
    x, theta = dataset.generate_save_dataset(sim_data_dir)
