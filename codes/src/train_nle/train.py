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

NSC_DIR = Path(__file__).resolve().parent.parent.parent.parent.as_posix()  # NSC dir
sys.path.append(f"{NSC_DIR}/codes/src")

from utils.train import print_cuda_info
from utils.setup import check_path, clean_cache
from utils.set_seed import setup_seed
from neural_nets.embedding_nets_p5 import GRU3_FC, Conv_LSTM, Conv_Transformer
from utils.dataset.dataset import update_prior_min_max


from neural_nets.my_likelihood_nn import my_likelihood_nn
from train_nle.MyLikelihoodEstimator import CNLE


class Solver:
    def __init__(self, config, store_config=True):
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
        self.l_theta = len(self.config["prior"]["prior_min"])

        if store_config:
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

    def get_neural_likelihood(self, iid_batch_size_x=2, iid_batch_size_theta=-1):
        # define embedding net
        embedding_net = nn.Identity()

        neural_likelihood = my_likelihood_nn(
            model="cnle",
            embedding_net=embedding_net,
            **dict(
                config=self.config,
                iid_batch_size_x=iid_batch_size_x,
            ),
        )

        return neural_likelihood

    def init_inference(
        self, iid_batch_size_x=2, iid_batch_size_theta=2, sum_writer=True
    ):
        """initialize inference

        iid_batch_size_x: used when doing the posterior inference

        """
        if sum_writer:
            writer = SummaryWriter(log_dir=str(self.log_dir))
        else:
            writer = None

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
        neural_likelihood = self.get_neural_likelihood(
            iid_batch_size_x=iid_batch_size_x,
            iid_batch_size_theta=iid_batch_size_theta,
        )
        self.inference = CNLE(
            prior=self.prior,
            density_estimator=neural_likelihood,
            device=self.device,
            logging_level="CRITICAL",
            summary_writer=writer,
            show_progress_bars=True,
        )

    def sbi_train(self, debug=False):
        # initialize inference
        self.init_inference()

        # initialize inference dataset (data will not be used)
        self.inference.append_simulations(
            theta=torch.randn(10, 4),
            x=torch.cat([torch.randn(10, 14), torch.randint(0, 2, (10, 1))], dim=1),
            data_device="cpu",
        )

        # run training
        self.inference, density_estimator = self.inference.train(
            config=self.config,
            prior_limits=self._get_limits(),
            continue_from_checkpoint=self.config.continue_from_checkpoint,
            debug=debug,
        )

        # # build MCMC posterior
        # mcmc_parameters = dict(
        #     warmup_steps=100,
        #     thin=10,
        #     num_chains=10,
        #     num_workers=10,
        #     init_strategy="sir",
        # )

        # cnle_posterior = self.inference.build_posterior(
        #     density_estimator=density_estimator,
        #     mcmc_parameters=mcmc_parameters,
        # )

        # # generate posterior samples
        # cnle_samples = cnle_posterior.sample(
        #     (num_samples,), x=x_o.reshape(num_trials, 2)
        # )


@hydra.main(
    config_path="../config_nle", config_name="config-nle-test", version_base=None
)
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
