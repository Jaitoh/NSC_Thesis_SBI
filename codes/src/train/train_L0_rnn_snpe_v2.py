"""
using cnn to parse input sequence of shape x(batch_size, D,M,S,T,C, L_x) theta(D,M,S,T,C, L_theta)
and output the probability of each base
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

from sbi import analysis
from sbi import utils as utils
from sbi.utils.get_nn_models import posterior_nn

from torch.utils.tensorboard import SummaryWriter

import sys

sys.path.append('./src')
# from dataset.dataset_generator import simulate_and_store, prepare_training_data_from_sampled_Rchoices
# from dataset.seqC_generator import seqC_generator
from config.load_config import load_config
from codes.src.dataset.dataset import training_dataset
from codes.src.simulator.seqC_generator import seqC_generator
from codes.src.dataset.simulate_for_sbi import simulate_for_sbi
from codes.src.utils.get_xo import get_xo
from utils.set_seed import setup_seed, seed_worker
from codes.src.train.collate_fn import collate_fn_probR
from neural_nets.embedding_nets import LSTM_Embedding
from inference.MySNPE_C import MySNPE_C
from codes.src.simulator.model_sim_pR import get_boxUni_prior
from utils.train import get_args, print_cuda_info

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
        print(f'using device: {self.device}')
        print_cuda_info(self.device)

        self.log_dir = Path(self.args.log_dir)
        self.data_dir = Path(config['data_dir'])
        self.sim_data_name = self.config['simulator']['save_name']
        self.train_data_name = self.config['dataset']['save_name']
        self._check_path()
        # save the config file using yaml
        yaml_path = Path(self.log_dir) / 'config.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f)
        print(f'config file saved to: {yaml_path}')

        # set seed
        seed = args.seed
        setup_seed(seed)
        self.g = torch.Generator()
        self.g.manual_seed(seed)

        self.prior              = None
        self.posterior          = None
        self.density_estimator  = None
        self.prior_max_train    = None
        self.prior_min_train    = None
        self.inference          = None

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


    def get_neural_posterior(self):

        # dms, l_x = x.shape[1], x.shape[2]
        d = len(self.config['x_o']['chosen_dur_list'])
        m = len(self.config['x_o']['chosen_MS_list'])
        s = self.config['x_o']['seqC_sample_per_MS']

        dms = d*m*s
        l_x = 15+1

        config_density = self.config['train']['density_estimator']

        embedding_net = LSTM_Embedding(
            dms         = dms,
            l           = l_x,
            hidden_size = config_density['embedding_net']['hidden_size'],
            output_size = config_density['embedding_net']['output_size'],
        )

        neural_posterior = posterior_nn(
            model           = config_density['posterior_nn']['model'],
            embedding_net   = embedding_net,
            hidden_features = config_density['posterior_nn']['hidden_features'],
            num_transforms  = config_density['posterior_nn']['num_transforms'],
        )

        return neural_posterior


    def sbi_train(self):
        """
        train the sbi model

        Args:
            x     (torch.tensor): shape (T*C, D*M*S, L_x)
            theta (torch.tensor): shape (T*C, L_theta)
        """
        # train the sbi model
        writer = SummaryWriter(log_dir=str(self.log_dir))

        # # observed data from trial experiment
        # x_o = get_xo(
        #     subject_id          = self.config['x_o']['subject_id'],
        #     chosen_dur_list     = self.config['x_o']['chosen_dur_list'],
        #     chosen_MS_list      = self.config['x_o']['chosen_MS_list'],
        #     seqC_sample_per_MS  = self.config['x_o']['seqC_sample_per_MS'],
        #     trial_data_path     = self.config['x_o']['trial_data_path'],
        #
        #     seqC_process_method = self.config['dataset']['seqC_process'],
        #     nan2num             = self.config['dataset']['nan2num'],
        #     summary_type        = self.config['dataset']['summary_type'],
        # )
        # self.x_o = torch.tensor(x_o, device=self.device)

        # prior
        self.prior_min_train = self.config['prior']['prior_min']
        self.prior_max_train = self.config['prior']['prior_max']

        prior = utils.torchutils.BoxUniform(
            low     = np.array(self.prior_min_train, dtype=np.float32),
            high    = np.array(self.prior_max_train, dtype=np.float32),
            device  = self.device,
        )
        self.prior = prior

        # get neural posterior
        neural_posterior = self.get_neural_posterior()

        self.inference = MySNPE_C(
            prior               = prior,
            density_estimator   = neural_posterior,
            device              = self.device,
            logging_level       = 'INFO',
            summary_writer      = writer,
            show_progress_bars  = True,
        )

        # print('---\ntraining property: ')
        print_cuda_info(self.device)
        start_time_total = time.time()
        self.density_estimator = []
        self.posterior = []
        proposal = prior
        training_config = self.config['train']['training']

        # dataloader kwargs
        Rchoice_method = self.config['dataset']['Rchoice_method']
        if Rchoice_method == 'probR':
            my_dataloader_kwargs = {
                'num_workers': training_config['num_workers'],
                'worker_init_fn':  seed_worker,
                'generator':   self.g,
                'collate_fn':  lambda batch:
                collate_fn_probR(
                    batch,
                    Rchoice_method=Rchoice_method,
                    num_probR_sample=self.config['dataset']['num_probR_sample'],
                ),
            }
        else:
            my_dataloader_kwargs = {
                'num_workers': training_config['num_workers'],
                'worker_init_fn':  seed_worker,
                'generator':   self.g,
            }
        if self.gpu:
            my_dataloader_kwargs['pin_memory'] = True


        # start training
        for run in range(training_config['num_runs']):

            print(f"======\nstart of run {run+1}/{training_config['num_runs']}\n======")

            # get simulated data
            x, theta = simulate_for_sbi(
                proposal        = proposal,
                config          = self.config,
                run             = run,
                save_sim_data   = self.config['simulator']['save_sim_data'],
                save_train_data = self.config['dataset']['save_train_data'],
            )

            theta = torch.tensor(theta, dtype=torch.float32) # avoid float64 error
            x     = torch.tensor(x,     dtype=torch.float32)

            print(f"---\nstart training")
            start_time = time.time()
            density_estimator = self.inference.append_simulations(
                theta    = theta,
                x        = x,
                proposal = proposal,
            ).train(
                num_atoms               = training_config['num_atoms'],
                training_batch_size     = training_config['training_batch_size'],
                learning_rate           = eval(training_config['learning_rate']),
                validation_fraction     = training_config['validation_fraction'],
                stop_after_epochs       = training_config['stop_after_epochs'],
                max_num_epochs          = training_config['max_num_epochs'],
                clip_max_norm           = training_config['clip_max_norm'],
                calibration_kernel      = None,
                resume_training         = training_config['resume_training'],
                force_first_round_loss  = training_config['force_first_round_loss'],
                discard_prior_samples   = training_config['discard_prior_samples'],
                use_combined_loss       = training_config['use_combined_loss'],
                retrain_from_scratch    = training_config['retrain_from_scratch'],
                show_train_summary      = training_config['show_train_summary'],
                dataloader_kwargs       = my_dataloader_kwargs,
            )  # density estimator

            print(f'finished training of run{run} in {(time.time()-start_time)/60:.2f} min')

            self.density_estimator.append(density_estimator)

            posterior = self.inference.build_posterior(density_estimator)
            self.check_posterior(posterior, run)
            self.posterior.append(posterior)

            # proposal = posterior.set_default_x(x_o)

        print(f"finished training of {training_config['num_runs']} runs in {(time.time()-start_time_total)/60:.2f} min")


    def save_model(self):

        print('---\nsaving model...')

        inference_dir           = self.log_dir / 'inference.pkl'
        density_estimator_dir   = self.log_dir / 'density_estimator.pkl'
        posterior_dir           = self.log_dir / 'posterior.pkl'

        with open(inference_dir, 'wb') as f:
            pickle.dump(self.inference, f)

        with open(density_estimator_dir, 'wb') as f:
            pickle.dump(self.density_estimator, f)

        with open(posterior_dir, 'wb') as f:
            pickle.dump(self.posterior, f)

        print('inference saved to: ',           inference_dir)
        print('density_estimator saved to: ',   density_estimator_dir)
        print('posterior saved to: ',           posterior_dir)

    def check_posterior(self, posterior, run):

        sampling_num = self.config['train']['posterior']['sampling_num']

        # for run in range(self.config['train']['training']['num_runs']):

        print(f'---\nchecking posterior of run {run}')

        start_time = time.time()
        samples = posterior.sample((sampling_num,), x=self.x_o)
        print(f'---\nfinished sampling in {(time.time()-start_time)/60:.2f} min')

        fig, axes = analysis.pairplot(
            samples =samples.cpu().numpy(),
            limits  =self._get_limits(),
            figsize =(10, 10),
            labels  =self.config['prior']['prior_labels'],
            upper=["kde"],
            diag=["kde"],
            # ticks=[[], []],
            # points=true_params.cpu().numpy(),
            # points_offdiag={'markersize': 5, 'markeredgewidth': 1},
            # points_colors='r',
        )

        save_path = self.log_dir / f'x_o_posterior_run{run}.png'
        fig.savefig(save_path)
        print(f'x_o_posterior_run{run} saved to: {save_path}')

    def _get_limits(self):
        return [[x, y] for x, y in zip(self.prior_min_train, self.prior_max_train)]

    def __decode_x_to_seqC(self, x):
        # x shape should be 1
        if len(x.shape) != 1:
            raise ValueError('x dimension should be 1')

        x = x[:-1]
        x = x[x != 0] # remove zeros
        nan2num = self.config['dataset']['nan2num']

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
            labels=self.config['prior']['prior_labels'],
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
                itertools.product(self.config['dataset']['train_data_dur_list'], self.config['dataset']['train_data_MS_list'])):

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
                labels=self.config['prior']['prior_labels'],
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

    print(f'---\nget args:')
    for arg, value in vars(args).items():
        print(f'{arg}: {value}')

    print('---\nconfig keys: ')
    print(config.keys())

    solver = Solver(args, config)
    solver.sbi_train()
    solver.save_model()
    solver.check_posterior()
    
    # # save the solver
    # with open(Path(args.log_dir) / 'solver.pkl', 'wb') as f:
    #     # pickle.dump(solver, f)
    #     dill.dump(solver, f)
    # print(f'solver saved to: {Path(args.log_dir) / "solver.pkl"}')


if __name__ == '__main__':
    main()

