""" training pipeline p4
features: offline hand-crafted features
training: one round
ignore the 2nd and 3rd parameter ss
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

from train.MyPosteriorEstimator_p4 import MySNPE_C_P4
from utils.train import print_cuda_info
from utils.setup import check_path, clean_cache
from utils.set_seed import setup_seed
from neural_nets.embedding_nets_p4 import (
    GRU_FC,
    Multi_Head_GRU_FC,
    GRU3_FC,
    LSTM3_FC,
    MLP_FC,
    Multi_Channel_CNN_FC,
    CNN_FC,
    CNN_FC2,
)
from utils.dataset.dataset import update_prior_min_max


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
        self.D = len(self.config.experiment_settings.chosen_dur_list)
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

    def get_neural_posterior(self):
        config_density = self.config.train.density_estimator
        concatenate_along_M = self.config.dataset.concatenate_along_M

        num_layers = 1
        input_size = 1 if concatenate_along_M else self.M
        hidden_size = config_density.embedding_net.hidden_size

        input_feature_length = np.sum(self.config.dataset.feature_lengths) * self.M
        print(f"\n=== embedding net === \n{config_density.embedding_net.type}")
        match config_density.embedding_net.type:
            case "mlp":
                embedding_net = MLP_FC(input_feature_length)

            case "cnn":
                embedding_net = CNN_FC(input_feature_length)

            case "cnn2":
                embedding_net = CNN_FC2(input_feature_length)

            case "mch_cnn":
                embedding_net = Multi_Channel_CNN_FC()

            case "gru_fc":
                embedding_net = GRU_FC(input_size, hidden_size, num_layers)

            case "gru3_fc":
                embedding_net = GRU3_FC(input_size, num_layers)

            case "lstm3_fc":
                embedding_net = LSTM3_FC(input_size, num_layers)

            case "multi_head_gru_fc":
                if not concatenate_along_M:  # [B, F1+F2+..., M]
                    feature_lengths = list(self.config.dataset.feature_lengths)
                else:  # [B, F1+F2+F3 + F1+F2+F3 + F1+F2+F3 ..., 1]
                    feature_lengths = self.M * list(self.config.dataset.feature_lengths)
                print(f"{len(feature_lengths)} heads")
                embedding_net = Multi_Head_GRU_FC(
                    feature_lengths, input_size, hidden_size, num_layers
                )

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

    def init_inference(self, ignore_ss=True):
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
        neural_posterior = self.get_neural_posterior()
        self.inference = MySNPE_C_P4(
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
        self.init_inference(ignore_ss=self.config.prior.ignore_ss)

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

        # save model
        # torch.save(
        #     deepcopy(density_estimator.state_dict()),
        #     f"{self.log_dir}/model/a_final_best_model_state_dict.pt",
        # )


@hydra.main(config_path="../config", config_name="config-p4-test", version_base=None)
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
