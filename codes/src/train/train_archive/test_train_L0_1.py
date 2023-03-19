import pickle

import torch
import numpy as np
from typing import Any, Callable
import sbi.inference

from sbi import utils as utils
import sys

sys.path.append('./src')

from data_generator.dataset_generator import prepare_training_data_from_sampled_Rchoices

import h5py
import shutil

# get arg input from command line
import argparse
import yaml
import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


def print_cuda_info(device):
    if device == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

        torch.cuda.memory_summary(device=None, abbreviated=False)


def check_method(method):
    try:
        method_fun: Callable = getattr(sbi.inference, method.upper())
    except AttributeError:
        raise NameError("Method not available. `method` must be one of 'SNPE', 'SNLE', 'SNRE'.")
    return method_fun


test = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')
# device='cpu'

print_cuda_info(device)
# ========== prepare dataset ==========
# load the dataset
dataset_dir = Path('../data/training_datasets/training_dataset_test.hdf5') \
    if test else Path('../data/training_datasets/training_dataset.hdf5')
dur_list = [15]  # if test else list(range(3, 16, 2))
nan2num = -1
num_probR_sample = 10  # 10 if test else 30
# check if the dataset exists
x_dir = Path('../data/training_datasets/x_test.pt') if test else Path('../data/training_datasets/x_15_1.pt')
theta_dir = Path('../data/training_datasets/theta_test.pt') if test else Path('../data/training_datasets/theta_15_1.pt')

# if not x_dir.exists():
print('Preparing training data...')
x, theta = prepare_training_data_from_sampled_Rchoices(
    dataset_dir=dataset_dir,
    dur_list=dur_list,
    nan2num=nan2num,
    num_probR_sample=num_probR_sample,
    part_of_seqC=0.08,  # 700 -> 56
    # part_of_seqC=0.08,  # 700 -> 56
    part_of_prior=0.2,  # 500 -> 100,
    remove_sigma2i=True,
)
# save x, theta
torch.save(x, x_dir)
torch.save(theta, theta_dir)
print('x, theta saved.')
# else:
#     print('x, theta found. Loading training data...')
#     start_time = time.time()
#     x = torch.load(x_dir)
#     theta = torch.load(theta_dir)
#     print('x, theta loaded. Time elapsed: {:.2f} s'.format(time.time() - start_time))

x, theta = x.to(device), theta.to(device)

# ========== train ==========
method = 'snpe'
method_fun = check_method(method)

log_dir = Path('./src/train/logs_15_p0/log_test') if test else Path('./src/train/logs_15_p0/log_sample_Rchoices1')
log_dir.mkdir(parents=True, exist_ok=True)  # create folder if the folder does not exist
# remove all files in the folder
for file in log_dir.glob('*'):
    file.unlink()
writer = SummaryWriter(log_dir=str(log_dir))

prior_min_train = [-3.7, 0, 0, -5]
prior_max_train = [2.5, 71, 18, 7]
prior_train = utils.torchutils.BoxUniform(
    low=torch.as_tensor(prior_min_train), high=torch.as_tensor(prior_max_train), device=device
)

inference = method_fun(prior=prior_train,
                       density_estimator='maf',
                       device=device,
                       logging_level='WARNING',
                       summary_writer=writer,
                       show_progress_bars=True,
                       )

print('\nstart training')
start_time = time.time()
print_cuda_info(device)
density_estimator = inference.append_simulations(theta=theta, x=x).train(
    num_atoms=10,
    training_batch_size=50,
    learning_rate=5e-4,
    validation_fraction=0.1,
    stop_after_epochs=20,
    max_num_epochs=2 ** 31 - 1,
    clip_max_norm=5.0,
    calibration_kernel=None,
    resume_training=False,
    force_first_round_loss=False,
    discard_prior_samples=False,
    use_combined_loss=False,
    retrain_from_scratch=False,
    show_train_summary=True,
    # dataloader_kwargs = {'shuffle': True,
    #                      'num_workers': 16,
    #                      'worker_init_fn':  seed_worker,
    #                      'generator':   self.g,
    #                      'pin_memory':  True},
)
print('finished training in {:.2f} min'.format((time.time() - start_time) / 60))
print_cuda_info(device)

# save density estimator to a pickle file
density_estimator_dir = log_dir / 'density_estimator_test.pkl' if test else log_dir / 'density_estimator.pkl'
inference_dir = log_dir / 'inference_test.pkl' if test else log_dir / 'inference.pkl'
posterior_dir = log_dir / 'posterior_test.pkl' if test else log_dir / 'posterior.pkl'

density_estimator.to('cpu')
posterior = inference.build_posterior(density_estimator)

# save objects to files
with open(density_estimator_dir, 'wb') as f:
    pickle.dump(density_estimator, f)
print('density_estimator saved to', density_estimator_dir)
with open(posterior_dir, 'wb') as f:
    pickle.dump(posterior, f)
print('posterior saved to', posterior_dir)
with open(inference_dir, 'wb') as f:
    pickle.dump(inference, f)
print('inference saved to', inference_dir)
