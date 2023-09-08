""" training pipeline p5
rebuild the pipeline of training
this time using network parsing the relationship between seqC and chR
"""
import os
import sys
import torch
import hydra
import numpy as np
from pathlib import Path
from copy import deepcopy
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from sbi import utils as utils
from sbi.utils.get_nn_models import posterior_nn

from pathlib import Path

NSC_DIR = Path(__file__).resolve().parent.parent.parent.parent.as_posix()  # NSC dir
sys.path.append(f"{NSC_DIR}/codes/src")

from train.MyPosteriorEstimator_p5 import MySNPE_C_P5
from utils.train import print_cuda_info
from utils.setup import check_path, clean_cache, adapt_path
from utils.set_seed import setup_seed
from neural_nets.embedding_nets_p5 import GRU3_FC, Conv_LSTM, Conv_Transformer, Conv_NET
from utils.dataset.dataset import update_prior_min_max


class Solver:
    def __init__(self, config, training_mode=True):
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
        if training_mode:
            yaml_path = adapt_path(Path(self.log_dir) / "config.yaml")
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

    def get_neural_posterior(self):
        config_density = self.config.train.density_estimator

        print(f"\n=== embedding net === \n{config_density.embedding_net.type}")
        match config_density.embedding_net.type:
            case "gru3_fc":
                embedding_net = GRU3_FC(self.DMS)

            case "conv_transformer":
                embedding_net = Conv_Transformer(self.DMS)

            case "conv_lstm":
                embedding_net = Conv_LSTM(self.DMS)

            case "conv_net":
                embedding_net = Conv_NET(self.DMS)

        neural_posterior = posterior_nn(
            model=config_density["posterior_nn"]["model"],
            embedding_net=embedding_net,
            hidden_features=config_density["posterior_nn"]["hidden_features"],
            num_transforms=config_density["posterior_nn"]["num_transforms"],
            num_components=10,
            z_score_theta=None,  # remove z_score
            z_score_x=None,  # remove z_score
        )

        return neural_posterior

    def init_inference(self, sum_writer=True):
        if sum_writer:
            writer = SummaryWriter(log_dir=str(self.log_dir))
        else:
            writer = None

        # if ingnore_ss not in config, set it to False
        if "ignore_ss" not in self.config.prior:
            self.config.prior.ignore_ss = False
        if "normalize" not in self.config.prior:
            self.config.prior.normalize = False

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
        neural_posterior = self.get_neural_posterior()
        self.inference = MySNPE_C_P5(
            prior=self.prior,
            density_estimator=neural_posterior,
            device=self.device,
            logging_level="INFO",
            summary_writer=writer,
            show_progress_bars=True,
        )

        return self.inference

    def sbi_train(self, debug=False):
        # initialize inference
        self.init_inference()

        # initialize inference dataset
        self.inference.append_simulations(
            theta=torch.empty(1),
            x=torch.empty(1),
            proposal=self.prior,
            data_device="cpu",
        )

        # run training
        self.inference, density_estimator = self.inference.train(  # type: ignore
            config=self.config,
            prior_limits=self._get_limits(),
            continue_from_checkpoint=self.config.continue_from_checkpoint,
            debug=debug,
        )

        # # save model
        # torch.save(
        #     deepcopy(density_estimator.state_dict()),
        #     f"{self.log_dir}/model/a_final_best_model_state_dict.pt",
        # )


@hydra.main(config_path="../config", config_name="config-p5-test", version_base=None)
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
