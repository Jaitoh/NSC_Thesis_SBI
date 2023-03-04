import torch
import numpy as np
from typing import Any, Callable
import sbi.inference

from sbi import utils as utils
import sys
sys.path.append('./src')

import h5py
import shutil

# get arg input from command line
import argparse
import yaml
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(description='Run DC')
    parser.add_argument('--run_test', action='store_true', help='test mode')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--config_n_dir', type=str, default=f"{sim_folder_dir}/src/trains/GSC/config/n0.yaml",  help='noise configuration file directory')
    parser.add_argument('--config_f_dir', type=str, default=f"{sim_folder_dir}/src/trains/GSC/config/f0p0.yaml",  help='feature configuration file directory')
    parser.add_argument('--config_m_dir', type=str, default=f"{sim_folder_dir}/src/trains/GSC/config/m0p0.yaml", help='model configuration file directory')
    parser.add_argument('--f_dir', type=str, default=f"{sim_folder_dir}/datasets/GSC_v2/simulated/", help='output feature file directory')
    if debug:
        parser.add_argument('--log_dir', type=str, default=f"{sim_folder_dir}/src/trains/GSC/logs/run_test/n0_f0p0_m0p0_s0_test", help='run recoding folder: e.g. n0_f0p0_m0p0_s0_test')
    else:
        parser.add_argument('--log_dir', type=str, default=f"{sim_folder_dir}/src/trains/GSC/logs/run_test/n0_f0p0_m0p0_s0", help='run recoding folder: e.g. n0_f0p0_m0p0_s0')
    args = parser.parse_args()
    return args

# get data
class Solver:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.gpu = self.args.gpu and torch.cuda.is_available()
        self.device = torch.device('cuda:0') if self.gpu else torch.device('cpu')
        
        self.logdir = Path('.') / 'runs' / self.args.name
        if self.logdir.exists() and not self.args.eval:
            if args.overwrite:
                shutil.rmtree(self.logdir)
            else:
                assert False, f'Run dir {str(self.logdir)} already exists.'

        self.build_dataset()
        self.build_model()

        # model to cuda
        if self.gpu:
            self.model = self.model.cuda()
            self.loss_fn = self.loss_fn.cuda()

        try:
            method_fun: Callable = getattr(sbi.inference, method.upper())
        except AttributeError:
            raise NameError(
                "Method not available. `method` must be one of 'SNPE', 'SNLE', 'SNRE'."
            )

    # simulator, prior = prepare_for_sbi(simulator, prior)

    inference = method_fun(prior=prior)

    print('start training')
    _ = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior()
    print('finished training')

    with h5py.File(save_data_dir, 'w') as f:
        f.create_dataset('posterior', data=posterior)
    print('posterior saved to .h5 file')

def main():
    args = get_args()
    config = yaml.safe_load(open(args.config, 'r'))
    solver = Solver(args, config)
    if args.eval:
        solver.evaluate()
    else:
        solver.solve()

if __name__ == '__main__':
    main()