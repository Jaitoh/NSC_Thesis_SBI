import torch
import numpy as np
from typing import Any, Callable
import sbi.inference

from sbi import utils as utils
import sys
sys.path.append('./src')
from data_generator.dataset_for_training import prepare_dataset_for_training
from data_generator.input_c import seqCGenerator

import h5py
import shutil

# get arg input from command line
import argparse
import yaml
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

def main(
    save_dir,
    gen_data=False,
    train=False,
         ):
    # args = get_args()
    # config = yaml.safe_load(open(args.config, 'r'))
    # solver = Solver(args, config)
    
    # model to cuda
    # if self.gpu:
    #     self.model = self.model.cuda()
    #     self.loss_fn = self.loss_fn.cuda()

    # ========== prepare dataset ==========
    data_dir = Path(f'{save_dir}/dataset.h5')
    with h5py.File(data_dir, 'w') as f:
        f.create_dataset('test', data='test')
    print('folder exists')
    
    if gen_data:
        # generate seqC input sequence
        seqC_MS_list = [0.2, 0.4, 0.8]
        seqC_dur_max = 14
        seqC_sample_size = 700*100
        seqC_sample_size = 700
        seqC_gen = seqCGenerator()
        seqCs = seqC_gen.generate(seqC_MS_list,
                                seqC_dur_max,
                                seqC_sample_size,
                                add_zero=True
                                )
        print(f'generated seqC shape', seqCs.shape)
        
        # generate prior distribution
        prior_min = [-3.7, -36, 0, -34, 5]
        prior_max = [2.5,  71,  0,  18, 7]
        prior = utils.torchutils.BoxUniform(
            low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
        )
        num_prior_sample = int(10**(len(prior_min)-1)) # 10000 in this case
        num_prior_sample = 1000
        num_prior_sample = 100
        print(f'prior sample size', num_prior_sample)
        
        # generate dataset
        model_name      = 'B-G-L0S-O-N-'
        nan2num         = -2
        num_LR_sample   = 10
        num_LR_sample   = 5
        
        x, theta = prepare_dataset_for_training(
            seqCs=seqCs,
            prior=prior,
            num_prior_sample=num_prior_sample,
            modelName=model_name,
            use_seqC_summary=False,
            summary_length=8,
            nan2num=nan2num,
            num_LR_sample=num_LR_sample,
            num_workers=-1
        )
        
        print('generated dataset shape', f'x: {x.shape}', f'theta: {theta.shape}')
        print('example of x and theta', x[0], theta[0])
        
        # save the dataset in a hdf5 file
        with h5py.File(data_dir, 'w') as f:
            f.create_dataset('x', data=x)
            f.create_dataset('theta', data=theta)
            f.create_dataset('seqCs', data=seqCs)
            f.create_dataset('prior', data=prior)
            
        print(f'dataset saved to {data_dir}')
        
    else:
        # load the dataset from a hdf5 file
        with h5py.File(data_dir, 'r') as f:
            x = f['x'][:]
            theta = f['theta'][:]
            seqCs = f['seqCs'][:]
            prior = f['prior'][:]
    
    if train:
        # train
        method = 'snpe'
        try:
            method_fun: Callable = getattr(sbi.inference, method.upper())
        except AttributeError:
            raise NameError("Method not available. `method` must be one of 'SNPE', 'SNLE', 'SNRE'.")
        
        # writer = SummaryWriter(log_dir=str(save_dir))
        inference = method_fun(prior=prior,
                                density_estimator='maf',
                                device='gpu',
                                logging_level='WARNING',
                                summary_writer=None,
                                show_progress_bars=True,
                                )
        # simulator, prior = prepare_for_sbi(simulator, prior)

        print('start training')
        _ = inference.append_simulations(theta, x).train(
            training_batch_size = 50,
            learning_rate = 5e-4,
            validation_fraction = 0.1,
            stop_after_epochs = 20,
            max_num_epochs = 2**31 - 1,
            clip_max_norm = 5.0,
            calibration_kernel = None,
            resume_training = False,
            force_first_round_loss = False,
            discard_prior_samples =  False,
            retrain_from_scratch =  False,
            show_train_summary =  True,
            # dataloader_kwargs = {'shuffle': True, 
            #                      'num_workers': 16, 
            #                      'worker_init_fn':  seed_worker, 
            #                      'generator':   self.g,
            #                      'pin_memory':  True},
        )
        posterior = inference.build_posterior()
        print('finished training')
        
        # posterior inference
        with h5py.File(save_dir, 'w') as f:
            f.create_dataset('posterior', data=posterior)
        print('posterior saved to .h5 file')


if __name__ == '__main__':
    save_dir = Path('../data/dataset_for_training')
    main(
        save_dir, 
        gen_data=True,
        train=True,
        )