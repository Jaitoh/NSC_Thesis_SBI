from pathlib import Path
import pickle
import scipy.io as sio
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import yaml
from tqdm import tqdm

import sys
sys.path.append('./src')
from utils.set_seed import setup_seed, seed_worker
setup_seed(0)
from dataset.dataset_pipeline import process_x_seqC_part

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
# from contextlib import contextmanager
# import os

# @contextmanager
# def suppress_stdout():
#     with open(os.devnull, "w") as devnull:
#         old_stdout = sys.stdout
#         sys.stdout = devnull
#         try:  
#             yield
#         finally:
#             sys.stdout = old_stdout

def load_merged_config(config_path):
    # load yaml config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def load_posterior(log_dir):
    inference_dir = log_dir / 'inference.pkl'
    dens_est_dir = log_dir / 'dens_est.pkl'
    posterior_dir = log_dir / 'posterior.pkl'

    # load trained inference, density estimator and posterior
    with open(inference_dir, 'rb') as f:
        inference = pickle.load(f)
    with open(dens_est_dir, 'rb') as f:
        density_estimator = pickle.load(f)
    with open(posterior_dir, 'rb') as f:
        posterior = pickle.load(f)
        
    density_estimator.to(device)
    posterior = inference.build_posterior(density_estimator, 
                                        #   device = device,
                                        # sample_with='rejection', # mcmc, rejection, vi
                                        )
    return density_estimator, posterior


def load_trial_data(trial_data_path, trial_len, subject_id): #observations
        
    trials = sio.loadmat(trial_data_path)
    # print(trials['info'][0,44])
    trial_data_len = trials['data'][0][1]
    trial_subject_id = trials['data'][0][44]
    trial_len_idx = np.where(trial_data_len == trial_len)[0]
    trial_subj_idx = np.where(trial_subject_id == subject_id)[0]
    trial_idx = np.intersect1d(trial_len_idx, trial_subj_idx)

    trial_data = trials['data'][0][0][trial_idx, :]
    trial_choice = trials['data'][0][42][trial_idx, :]
    # trial_data = np.concatenate((trial_data, trial_choice), axis=1)
    # trial_data = torch.tensor(trial_data).float().to(device)
    
    return trial_data, trial_choice
    
def posterior_sampling(obs_data, n_sub_samples, posterior, posterior_sampling_save_path=None):
    samples = torch.zeros((obs_data.shape[0]*n_sub_samples, 4)).to(device)
    print(obs_data.device)
    for i in tqdm(range(obs_data.shape[0])):
    # for i in range(10):
        # print(f'generating samples for trial {i+1}/{obs_data.shape[0]}')
        # with suppress_stdout():
        samples_ = posterior.sample((n_sub_samples,), x=obs_data[i,:],show_progress_bars=False)
        samples[i*n_sub_samples:(i+1)*n_sub_samples, :] = samples_
    
    if posterior_sampling_save_path != None:
        with open(posterior_sampling_save_path, 'wb') as f:
            pickle.dump(samples, f)
    
    posterior_samples = samples
    return posterior_samples


class posterior_inference:
    def __init__(self, 
                 config_path, 
                 log_dir, 
                 trial_data_path, 
                 trial_len, 
                 subject_id, 
                 n_sub_samples,
                 ):
        self.n_sub_samples = n_sub_samples
        
        config = load_merged_config(config_path)
        print(config['infer'].keys())
        self.prior_min = config['infer']['prior_min_train']
        self.prior_max = config['infer']['prior_max_train']
        self.prior_labels = config['infer']['prior_labels']

        density_estimator, posterior = load_posterior(log_dir)
        
        # load participant trial data
        print(f'loading and processing trial data from {trial_data_path} for subject {subject_id} with trial length {trial_len}...')
        trial_data, trial_choice = load_trial_data(trial_data_path, trial_len, subject_id)
        # print(f'trial_data {trial_data.shape}: {trial_data}, trial_choice {trial_choice.shape}: {trial_choice}')
        obs_data = process_x_seqC_part(trial_data, config)
        obs_data = np.concatenate((obs_data, trial_choice), axis=1)
        obs_data = torch.tensor(obs_data).float().to(device)
        self.obs_data = obs_data
        self.trial_choice = trial_choice
        self.trial_data = trial_data
        
        posterior_sampling_save_path = log_dir / 'samples.pkl'
        print(f'generating posterior samples...')
        self.posterior_samples = posterior_sampling(obs_data, n_sub_samples, posterior, posterior_sampling_save_path)
        print(f'posterior samples saved to {posterior_sampling_save_path}')
        
    def _plot_hist(self, samples_sub, axs, num_bins=50):
        
        ax = axs[0][0]
        ax.hist(samples_sub[:,0].numpy(), bins=num_bins, color='tab:blue')
        ax.set_title('bias')
        ax.set_xlim(self.prior_min[0], self.prior_max[0])
        ax = axs[0][1]
        ax.hist(samples_sub[:,1].numpy(), bins=num_bins, color='tab:blue')
        ax.set_title('sigma2a')
        ax.set_xlim(self.prior_min[1], self.prior_max[1])
        ax = axs[1][0]
        ax.hist(samples_sub[:,2].numpy(), bins=num_bins, color='tab:blue')
        ax.set_title('sigma2s')
        ax.set_xlim(self.prior_min[2], self.prior_max[2])
        ax = axs[1][1]
        ax.hist(samples_sub[:,3].numpy(), bins=num_bins, color='tab:blue')
        ax.set_title('L0')
        ax.set_xlim(self.prior_min[3], self.prior_max[3])
        
    def generate_one_hist(self, obs_data_idx=0):

        samples_sub = self.posterior_samples[self.n_sub_samples*obs_data_idx:self.n_sub_samples*(obs_data_idx+1),:]
        fig, axs = plt.subplots(2,2, figsize=(15,10))
        self._plot_hist(samples_sub, axs)

    def _update(self, frame):
        print(f'{frame}/{self.obs_data.shape[0]}', end='\r')
        
        samples_sub = self.posterior_samples[:self.n_sub_samples*frame,:]
        self._plot_hist(samples_sub, self.axs)
    
    def gen_animation(self, ani_name, total_frames=100):

        fig, self.axs = plt.subplots(2,2, figsize=(15, 10))
        ani = animation.FuncAnimation(fig, self._update, frames=np.linspace(0, self.obs_data.shape[0], total_frames, dtype=np.int), interval=5)
        # ani = animation.FuncAnimation(fig, update, frames=10, interval=2)
        animation_location = Path('./notebook/figures/')
        ani.save(animation_location/ani_name, fps=5)
        

if __name__ == '__main__':
    
    experiment_list = ['b1', 'b3', 'b4', 'b5', 'b6', 'c1', 'c5', 'c6', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6']
    # for experiement_id in experiment_list:
    for experiement_id in ['c5']:
        print(f'generating animation for experiment {experiement_id}...')
        subject_id=5
        trial_len=15 
        n_sub_samples=5000
        
        trial_data_path='../data/trials.mat'
        log_dir = Path(f'./src/train/logs/logs_L0_v1/log-train_L0-{experiement_id}')
        ani_name = f'ani_{experiement_id}_subj_{subject_id}_trial_len_{trial_len}.mp4'
        config_path = log_dir / 'config.yaml'

        P = posterior_inference(config_path, 
                        log_dir, 
                        trial_data_path, 
                        trial_len, 
                        subject_id, 
                        n_sub_samples,
                        )
        
        P.gen_animation(ani_name, total_frames=10)
        