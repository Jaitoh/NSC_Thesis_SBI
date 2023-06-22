""" training pipeline p4
features: offline hand-crafted features
training: one round
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

from sbi.utils.get_nn_models import posterior_nn

sys.path.append("./src")
from utils.train import print_cuda_info
from utils.setup import check_path, clean_cache
from utils.set_seed import setup_seed, seed_worker


class Solver:
    def __init__(self, config):
        self.config = config

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
        # self.l_x = 15 + 1 TODO:

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

        # TODO: embedding_net
        embedding_net = net()

        neural_posterior = posterior_nn(
            model=config_density["posterior_nn"]["model"],
            embedding_net=embedding_net,  # type: ignore
            hidden_features=config_density["posterior_nn"]["hidden_features"],
            num_transforms=config_density["posterior_nn"]["num_transforms"],
        )

        return neural_posterior

    def get_my_data_kwargs(self):
        config_dataset = self.config.dataset
        prefetch_factor = config_dataset.prefetch_factor

        return {
            "num_workers": config_dataset.num_workers,
            "worker_init_fn": seed_worker,
            "prefetch_factor": prefetch_factor + 2,
            "collate_fn": lambda batch: collate_fn(  # TODO: collate_fn
                batch=batch,
                config=self.config,
                shuffling_method=config_dataset.shuffling_method,
            ),
        }

    def sbi_train(self, debug=False):
        writer = SummaryWriter(log_dir=str(self.log_dir))

        # prior
        self.prior_min = self.config.prior.prior_min
        self.prior_max = self.config.prior.prior_max
        self.prior = utils.torchutils.BoxUniform(  # type: ignore
            low=np.array(self.prior_min, dtype=np.float32),
            high=np.array(self.prior_max, dtype=np.float32),
            device=self.device,
        )

        # get neural posterior
        neural_posterior = self.get_neural_posterior()
        MySNPE = MySNPE_C_P4  # TODO: MySNPE_C_P4
        self.inference = MySNPE(
            prior=self.prior,
            density_estimator=neural_posterior,
            device=self.device,
            logging_level="INFO",
            summary_writer=writer,
            show_progress_bars=True,
        )

        # dataloader kwargs, initialize inference dataset
        my_dataloader_kwargs = self.get_my_data_kwargs()
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
            dataloader_kwargs=my_dataloader_kwargs,
            continue_from_checkpoint=self.config.continue_from_checkpoint,
            debug=debug,
        )

        # save model
        torch.save(
            deepcopy(density_estimator.state_dict()),
            f"{self.log_dir}/model/a_final_best_model_state_dict.pt",
        )


@hydra.main(config_path="../config", config_name="config-test", version_base=None)
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


if __name__ == "__main__":
    main()
