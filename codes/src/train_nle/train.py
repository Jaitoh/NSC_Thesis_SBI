import os
import sys
import hydra
import torch
from torch import nn
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from sbi import utils as utils
from sbi.utils.get_nn_models import posterior_nn
import sys

sys.path.append(Path(__file__).resolve().parent.parent.as_posix())  # src dir

from utils.train import print_cuda_info
from utils.setup import check_path, clean_cache
from utils.set_seed import setup_seed
from neural_nets.embedding_nets_p5 import GRU3_FC, Conv_LSTM, Conv_Transformer
from utils.dataset import update_prior_min_max


from neural_nets.my_nn_models import my_likelihood_nn
from train_nle.MyLikelihoodEstimator import CNLE

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.as_posix()  # NSC dir


class Solver:
    def __init__(self, config):
        self.config = config
        self.log_dir = config.log_dir
        # gpu info
        self.gpu = self.config.gpu and torch.cuda.is_available()
        self.device = "cuda" if self.gpu else "cpu"
        print(f"using device: {self.device}")
        print_cuda_info(self.device)

        # get experiement settings
        self.D = len(self.config.dataset.chosen_dur_list)
        self.M = len(self.config.experiment_settings.chosen_MS_list)
        self.S = self.config.experiment_settings.seqC_sample_per_MS
        self.DMS = self.D * self.M * self.S
        self.l_theta = len(self.config["prior"]["prior_min"])

        # save the config file using yaml
        yaml_path = Path(self.log_dir) / "config.yaml"
        with open(yaml_path, "w") as f:
            f.write(OmegaConf.to_yaml(config))
        print(f"config file saved to: {yaml_path}")

        # set seed
        self.seed = config.seed
        setup_seed(self.seed)

    def _get_limits(self):
        if self.prior_min is None or self.prior_max is None:
            return []
        return [[x, y] for x, y in zip(self.prior_min, self.prior_max)]

    def get_neural_likelihood(self):
        # config_density = self.config.train.density_estimator

        # print(f"\n=== embedding net === \n{config_density.embedding_net.type}")
        # match config_density.embedding_net.type:
        #     case "gru3_fc":
        #         embedding_net = GRU3_FC(self.DMS)

        #     case "conv_transformer":
        #         embedding_net = Conv_Transformer(self.DMS)

        #     case "conv_lstm":
        #         embedding_net = Conv_LSTM(self.DMS)

        #     # default case
        #     case _:
        #         embedding_net = nn.Identity()

        embedding_net = nn.Identity()

        neural_likelihood = my_likelihood_nn(
            model="cnle",
            embedding_net=embedding_net,
            **dict(
                log_transform_x=False,
                num_bins=5,
                num_transforms=2,
                tail_bound=10.0,
                hidden_layers=1,
                hidden_features=10,
            ),
        )

        return neural_likelihood

    def init_inference(self, ignore_ss=False):
        writer = SummaryWriter(log_dir=str(self.log_dir))

        # prior
        (
            self.prior_min,
            self.prior_max,
            unnormed_prior_min,
            unnormed_prior_max,
        ) = update_prior_min_max(
            prior_min=self.config.prior.prior_min,
            prior_max=self.config.prior.prior_max,
            ignore_ss=self.config.prior.ignore_ss,
            normalize=self.config.prior.normalize,
        )

        self.prior = utils.torchutils.BoxUniform(  # type: ignore
            low=np.array(self.prior_min, dtype=np.float32),
            high=np.array(self.prior_max, dtype=np.float32),
            device=self.device,
        )

        if self.config.prior.normalize:
            print(f"prior min before norm: {unnormed_prior_min}")
            print(f"prior max before norm: {unnormed_prior_max}")
        print(f"prior min: {self.prior_min}")
        print(f"prior max: {self.prior_max}")

        # get neural posterior
        neural_likelihood = self.get_neural_likelihood()
        self.inference = CNLE(
            prior=self.prior,
            density_estimator=neural_likelihood,
            device=self.device,
            logging_level="Error",
            summary_writer=writer,
            show_progress_bars=True,
        )

    def sbi_train(self, debug=False):
        # initialize inference
        self.init_inference(ignore_ss=self.config.prior.ignore_ss)

        # initialize inference dataset
        self.inference.append_simulations(
            theta=torch.randn(10, 4),
            x=torch.cat([torch.randn(10, 14), torch.randint(0, 2, (10, 1))], dim=1),
            data_device="cpu",
        )

        # run training
        self.inference, density_estimator = self.inference.train()

        # build MCMC posterior
        mcmc_parameters = dict(
            warmup_steps=100,
            thin=10,
            num_chains=10,
            num_workers=10,
            init_strategy="sir",
        )

        cnle_posterior = self.inference.build_posterior(
            density_estimator=density_estimator,
            mcmc_parameters=mcmc_parameters,
        )

        # generate posterior samples
        cnle_samples = cnle_posterior.sample(
            (num_samples,), x=x_o.reshape(num_trials, 2)
        )


@hydra.main(config_path="../config", config_name="config-nle-test", version_base=None)
def main(config: DictConfig):
    PID = os.getpid()
    print(f"PID: {PID}")

    log_dir = Path(config.log_dir)
    data_path = Path(config.data_path)
    check_path(log_dir, data_path)

    solver = Solver(config)
    solver.sbi_train(debug=config.debug)

    del solver
    clean_cache()
    print(f"PID: {PID} finished")


if __name__ == "__main__":
    main()
