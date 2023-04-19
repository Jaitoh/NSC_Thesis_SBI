"""

"""
import itertools
import pickle
import yaml
# import dill

# import h5py
# import yaml
# import glob
import argparse
import torch
import os
import numpy as np
import random
from pathlib import Path
import time
import shutil
from typing import Any, Callable
import sbi.inference
from sbi import analysis
from sbi import utils as utils

from torch.utils.tensorboard import SummaryWriter

import sys

sys.path.append('./src')
# from dataset.dataset_generator import simulate_and_store, prepare_training_data_from_sampled_Rchoices
# from dataset.seqC_generator import seqC_generator
from config.load_config import load_config
from codes.src.simulator.model_sim_pR import DM_simulate_and_store, _one_DM_simulation, _one_DM_simulation_and_output_figure
from codes.src.dataset.dataset import training_dataset
from codes.src.simulator.seqC_generator import seqC_generator

def get_args():
    """
    Returns:
        args: Arguments
    """
    parser = argparse.ArgumentParser(description='pipeline for sbi')
    # parser.add_argument('--run_test', action='store_true', help="")
    parser.add_argument('--seed', type=int, default=0, help="")
    parser.add_argument('--run_simulator', type=int, default=0, help="""run simulation to generate dataset and store to local file
                                                                        0: no simulation, load file directly and do the training
                                                                        1: run simulation and do the training afterwards
                                                                        2: only run the simulation and do not train""")
    parser.add_argument('--config_simulator_path', type=str, default="./src/config/test_simulator.yaml",
                        help="Path to config_simulator file")
    parser.add_argument('--config_dataset_path', type=str, default="./src/config/test_dataset.yaml",
                        help="Path to config_train file")
    parser.add_argument('--config_train_path', type=str, default="./src/config/test_train.yaml",
                        help="Path to config_train file")
    # parser.add_argument('--data_dir', type=str, default="../data/train_datas/",
    #                     help="simulated data store/load dir")
    parser.add_argument('--log_dir', type=str, default="./src/train/log_test", help="training log dir")
    parser.add_argument('--gpu', action='store_true', help='Use GPU.')
    # parser.add_argument('--finetune', type=str, default=None, help='Load model from this job for finetuning.')
    parser.add_argument('--eval', action='store_true', help='Evaluation mode.')
    parser.add_argument('-y', '--overwrite', action='store_true', help='Overwrite log dir.')
    args = parser.parse_args()

    return args


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
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def print_cuda_info(device):
    """

    Args:
        device: 'cuda' or 'cpu'
    """
    if device == 'cuda':
        print('--- CUDA info ---')
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


class Solver:
    """
        Solver for training sbi
    """

    def __init__(self, args, config):

        self.args = args
        self.config = config
        # self.test = self.args.run_test

        self.gpu = self.args.gpu and torch.cuda.is_available()
        # self.device = torch.device('cuda') if self.gpu else torch.device('cpu')
        self.device = 'cuda' if self.gpu else 'cpu'
        print_cuda_info(self.device)

        self.log_dir = Path(self.args.log_dir)
        self.data_dir = Path(config['data_dir'])
        # self.log_dir = self.log_dir.parent / 'log_test' if self.test else self.log_dir
        # self.sim_data_name = 'sim_data_test.h5' if self.test else self.config['simulator']['save_name']
        # self.train_data_name = 'train_data_test.h5' if self.test else self.config['train_data']['save_name']
        self.sim_data_name = self.config['simulator']['save_name']
        self.train_data_name = self.config['train_data']['save_name']
        self._check_path()
        # save the config file using yaml
        yaml_path = Path(self.log_dir) / 'config.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f)
        print(f'config file saved to: {yaml_path}')

    def _check_path(self):
        """
        check the path of log_dir and data_dir
        """

        print(f'---\nlog dir: {str(self.log_dir)}')
        print(f'data dir: {str(self.data_dir)}')

        # check log path: if not exists, create; if exists, remove or a fatal error
        if not self.log_dir.exists():
            os.makedirs(str(self.log_dir))
        elif self.log_dir.exists() and not self.args.eval:
            if self.args.overwrite:
                shutil.rmtree(self.log_dir)
                print(f'Run dir {str(self.log_dir)} emptied.')
                os.makedirs(str(self.log_dir))
            else:
                assert False, f'Run dir {str(self.log_dir)} already exists.'

        # check data path exists
        if not Path(self.data_dir).exists():
            assert False, f'Data dir {str(self.data_dir)} does not exist.'

    def get_sim_data(self):
        # obtain the whole dataset prepared by simulator

        config = self.config
        save_data_path = Path(self.data_dir) / self.sim_data_name

        DM_simulate_and_store(
            save_data_dir=save_data_path,

            seqC_MS_list=config['seqC']['MS_list'],
            seqC_dur_max=config['seqC']['dur_max'],  # 2, 4, 6, 8, 10, 12, 14 -> 7
            seqC_sample_size=config['seqC']['sample'],

            prior_min=config['prior']['prior_min'],
            prior_max=config['prior']['prior_max'],
            num_prior_sample=config['prior']['num_prior_sample'],

            model_name=config['simulator']['model_name'],
        )


    def get_dataset(self):

        # adapted the datset and choose pattern of interest for further training
        dataset = training_dataset(self.config)
        x, theta = dataset.generate_save_dataset(self.data_dir)

        return x.to(self.device), theta.to(self.device)

    def _evaluate_prior(self):
        # evealuate the prior
        prior_min = self.config['prior']['prior_min'].copy()
        prior_max = self.config['prior']['prior_max'].copy()
        assert self.prior_min_train == prior_min[:2] + prior_min[
                                                       2 + 1:], f"prior_min_train: {self.prior_min_train}, prior_min: {prior_min}"
        assert self.prior_max_train == prior_max[:2] + prior_max[
                                                       2 + 1:], f"prior_max_train: {self.prior_max_train}, prior_max: {prior_max}"

    def sbi_train(self, x, theta):
        # train the sbi model
        method = self.config['infer']['method']
        method_fun = check_method(method)

        writer = SummaryWriter(log_dir=str(self.log_dir))

        self.prior_min_train = self.config['infer']['prior_min_train']
        self.prior_max_train = self.config['infer']['prior_max_train']
        self._evaluate_prior()
        prior_train = utils.torchutils.BoxUniform(
            low=torch.as_tensor(self.prior_min_train), high=torch.as_tensor(self.prior_max_train), device=self.device
        )

        self.prior_train = prior_train
        self.prior_simulator = utils.torchutils.BoxUniform(
            low=torch.as_tensor(self.config['prior']['prior_min']), high=torch.as_tensor(self.config['prior']['prior_max']), device=self.device
        )

        self.inference = method_fun(prior=prior_train,
                                    density_estimator='maf',
                                    device=self.device,
                                    logging_level='INFO',
                                    summary_writer=writer,
                                    show_progress_bars=True,
                                    )

        print('---\ntraining property: ')
        print('training data shape: ', x.shape)
        print('training theta shape: ', theta.shape)
        print('method: ', method)
        print('prior min ', self.prior_min_train)
        print('prior max ', self.prior_max_train)
        print('---\nstart training...')

        start_time = time.time()
        print_cuda_info(self.device)

        infer_config = self.config['infer']
        # print(infer_config['learning_rate']-1)
        dens_est = self.inference.append_simulations(theta=theta, x=x).train(
            num_atoms=infer_config['num_atoms'],
            training_batch_size=infer_config['training_batch_size'],
            learning_rate=eval(infer_config['learning_rate']),
            validation_fraction=infer_config['validation_fraction'],
            stop_after_epochs=infer_config['stop_after_epochs'],
            max_num_epochs=infer_config['max_num_epochs'],
            clip_max_norm=infer_config['clip_max_norm'],
            calibration_kernel=None,
            resume_training=infer_config['resume_training'],
            force_first_round_loss=infer_config['force_first_round_loss'],
            discard_prior_samples=infer_config['discard_prior_samples'],
            use_combined_loss=infer_config['use_combined_loss'],
            retrain_from_scratch=infer_config['retrain_from_scratch'],
            show_train_summary=infer_config['show_train_summary'],
            # dataloader_kwargs = {'shuffle': True,
            #                      'num_workers': 16,
            #                      'worker_init_fn':  seed_worker,
            #                      'generator':   self.g,
            #                      'pin_memory':  True},
        )  # density estimator
        print('finished training in {:.2f} min'.format((time.time() - start_time) / 60))
        print_cuda_info(self.device)

        # self.dens_est = dens_est.to('cpu')
        self.dens_est = dens_est
        self.posterior = self.inference.build_posterior(self.dens_est)
        

    def save_model(self):
        print('---\nsaving model...')
        inference_dir = self.log_dir / 'inference.pkl'
        dens_est_dir = self.log_dir / 'dens_est.pkl'
        posterior_dir = self.log_dir / 'posterior.pkl'

        with open(inference_dir, 'wb') as f:
            pickle.dump(self.inference, f)
        print('inference saved to: ', inference_dir)

        with open(dens_est_dir, 'wb') as f:
            pickle.dump(self.dens_est, f)
        print('density_estimator saved to: ', dens_est_dir)

        with open(posterior_dir, 'wb') as f:
            pickle.dump(self.posterior, f)
        print('posterior saved to: ', posterior_dir)

    def _get_limits(self):
        return [[x, y] for x, y in zip(self.prior_min_train, self.prior_max_train)]

    def __decode_x_to_seqC(self, x):
        # x shape should be 1
        if len(x.shape) != 1:
            raise ValueError('x dimension should be 1')

        x = x[:-1]
        x = x[x != 0] # remove zeros
        nan2num = self.config['train_data']['nan2num']

        seqC = nan2num + x*(1-nan2num)

        return seqC

    def _check_posterior_seen_one(self, x, theta, truth_idx=300, sample_num=1000):

        true_params = theta[truth_idx, :]
        print(f'->\nevaluate with input x[{truth_idx},:]: {x[truth_idx, :]}')
        print(f'evaluation true params: {true_params}')

        seqC = self.__decode_x_to_seqC(x[truth_idx, :])
        model_name = self.config['simulator']['model_name']
        figure_name = f'seqC: {seqC}\nparams: {true_params}\nmodel: {model_name}'
        # insert 0 in the 2nd position of the tensor
        device = true_params.device
        params = torch.cat((true_params[:2], torch.zeros(1).to(device), true_params[2:]))
        simulator_output = _one_DM_simulation_and_output_figure(seqC.cpu().numpy(), params.cpu().numpy(), model_name, figure_name)
        (seqC, params, probR, fig) = simulator_output

        save_path = self.log_dir / f'seen_data_idx_{truth_idx}.png'
        fig.savefig(save_path, dpi=300)
        print('data simulation result saved to: ', save_path)
        
        samples = self.posterior.sample((sample_num,), x=x[truth_idx, :])

        fig, axes = analysis.pairplot(
            samples.cpu().numpy(),
            limits=self._get_limits(),
            # ticks=[[], []],
            figsize=(10, 10),
            points=true_params.cpu().numpy(),
            points_offdiag={'markersize': 5, 'markeredgewidth': 1},
            points_colors='r',
            labels=self.config['infer']['prior_labels'],
        )
        save_path = self.log_dir / f'seen_data_idx_{truth_idx}_posterior.png'
        fig.savefig(save_path)
        print(f'posterior_with_seen_data saved to: {save_path}')

    def check_posterior_seen(self, x, theta, truth_idx, sample_num=1000):
        """
        check the posterior with
            - trained data x -> theta distribution
            - unseen data x -> theta distribution

        Returns:
            distribution plot for each dimension of theta
        """

        if isinstance(truth_idx, list):
            for idx in truth_idx:
                self._check_posterior_seen_one(x, theta, idx, sample_num)
        else:
            self._check_posterior_seen_one(x, theta, truth_idx, sample_num)

    def _from_1seqC_to_1x(self, MS, dur):

        seqC = seqC_generator().generate(
                MS_list=[MS],
                sample_size=1,
                single_dur=dur,
            )[0,:]

        params = self.prior_simulator.sample((1,)).cpu().numpy()[0]
        model_name = self.config['simulator']['model_name']
        # simulator_output = _one_DM_simulation((seqC, params, model_name))
        figure_name = f'seqC:{seqC}\nmodel{model_name}\nparams:{params}'
        simulator_output = _one_DM_simulation_and_output_figure(seqC, params, model_name, figure_name)
        seqC, params, probR, fig = simulator_output

        dataset = training_dataset(self.config)
        x_seqC, theta, x_R = dataset.data_process_pipeline(seqC, params, probR)
        x_new = np.concatenate((x_seqC, x_R), axis=1)

        return x_new, fig

    def check_posterior_unseen(self, x, sample_num=1000):

        # probR sampling method
        for i, (dur, MS) in enumerate(
                itertools.product(self.config['train_data']['train_data_dur_list'], self.config['train_data']['train_data_MS_list'])):

            # num_unseen_sample = 1

            x_new, fig = self._from_1seqC_to_1x(MS, dur)
            while np.any(np.all(x.cpu().numpy() == x_new, axis=1)):
                x_new, fig = self._from_1seqC_to_1x(MS, dur)

            save_path = self.log_dir / f'unseen_data_dur{dur}_MS{MS}_{i}.png'
            fig.savefig(save_path, dpi=300)
            print('simulation result saved to: ', save_path)

            samples = self.posterior.sample((sample_num,), x=torch.from_numpy(x_new).float().to(self.device))
            fig, axes = analysis.pairplot(
                samples.cpu().numpy(),
                limits=self._get_limits(),
                # ticks=[[], []],
                figsize=(10, 10),
                # points=true_params,
                # points_offdiag={'markersize': 10, 'markeredgewidth': 1},
                # points_colors='r',
                labels=self.config['infer']['prior_labels'],
            )
            save_path = self.log_dir / f'unseen_data_dur{dur}_MS{MS}_{i}_posterior.png'
            fig.savefig(save_path)
            print(f'posterior_with_unseen_data saved to: {save_path}')


def main():
    args = get_args()

    config = load_config(
        config_simulator_path=args.config_simulator_path,
        config_dataset_path=args.config_dataset_path,
        config_train_path=args.config_train_path,
    )

    setup_seed(args.seed)
    
    print(f'---\nget args:')
    for arg, value in vars(args).items():
        print(f'{arg}: {value}')

    print('---\nconfig keys: ')
    print(config.keys())

    solver = Solver(args, config)
    
    if args.run_simulator == 0: # do not run the simulation, do training
        x, theta = solver.get_dataset()
        solver.sbi_train(x=x, theta=theta)
        solver.save_model()
        
    elif args.run_simulator == 2: # only run the simulation, do not train
        solver.get_sim_data()
        
    else: # run the simulation, do the training
        solver.get_sim_data()
    
        x, theta = solver.get_dataset()
        solver.sbi_train(x=x, theta=theta)
        solver.save_model()

    # save the solver
    # with open(Path(args.log_dir) / 'solver.pkl', 'wb') as f:
    #     # pickle.dump(solver, f)
    #     dill.dump(solver, f)
    # print(f'solver saved to: {Path(args.log_dir) / "solver.pkl"}')

    print('---\ncheck posterior with seen data (randomly choose 10 data points to check posterior):')
    # randomly choose 10 data points to check posterior
    random_idxes = np.random.randint(0, x.shape[0], size=10)
    solver.check_posterior_seen(x, theta, truth_idx=list(random_idxes), sample_num=5000)

    print('---\n\n>>>check posterior with unseen data:')
    solver.check_posterior_unseen(x, sample_num=5000)


if __name__ == '__main__':
    main()
