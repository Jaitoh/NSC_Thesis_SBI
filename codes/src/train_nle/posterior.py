"""check posterior for the trained model"""
import os
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize
from omegaconf import OmegaConf

import sys
from pathlib import Path

NSC_DIR = Path(__file__).resolve().parent.parent.parent.parent.as_posix()  # NSC dir
sys.path.append(f"{NSC_DIR}/codes/src")

from train_nle.train import Solver
from train_nle.MyLikelihoodEstimator import MyLikelihoodEstimator
from utils.setup import check_path, clean_cache
from utils.train import WarmupScheduler, plot_posterior_with_label, load_net


# @hydra.main(config_path="../config", config_name="config-nle-test", version_base=None)
# def main(config: DictConfig):
hydra.core.global_hydra.GlobalHydra.instance().clear()
initialize(config_path="../config", job_name="test_nle")
config = compose(config_name="config-nle-test")
print(OmegaConf.to_yaml(config))

model_path = (
    Path(NSC_DIR) / "codes/src/train_nle/logs/nle-lstm/model/model_check_point.pt"
)


PID = os.getpid()
print(f"PID: {PID}")

log_dir = Path(config.log_dir)
data_path = Path(config.data_path)

# initialize solver and network
solver = Solver(config)
solver.init_inference()
(
    _,
    _,
    density_estimator,
    train_dataset,
    valid_dataset,
) = solver.inference.prepare_dataset_network(
    config,
    continue_from_checkpoint=model_path,
    device="cuda" if torch.cuda.is_available() and config.gpu else "cpu",
)

# build posterior
mcmc_parameters = dict(
    warmup_steps=100,
    thin=10,
    num_chains=10,
    num_workers=10,
    init_strategy="sir",
)

posterior = solver.inference.build_posterior(
    density_estimator=density_estimator,
    prior=solver.inference._prior,
    sample_with="mcmc",
    mcmc_method="slice",
    mcmc_parameters=mcmc_parameters,
    # vi_method="rKL",
    # vi_parameters={},
    # rejection_sampling_parameters={},
)


# prepare data for posterior
def get_data_for_theta(train_dataset, idx_theta, chR=True):
    """
    get the data from the dataset for a given theta
    """
    theta_value = train_dataset.theta_all[idx_theta]

    if chR:
        # train_dataset.seqC_all.shape  # [MS, 15]
        # train_dataset.chR_all.shape  # [MS, T, C, 1]
        # train_dataset.theta_all.shape  # [T, 4]
        n_seq = train_dataset.seqC_all.shape[0]
        T, C = train_dataset.chR_all.shape[1:3]
        train_data = torch.empty((n_seq, C, 15))  # [MS, C, 15]
        for idx_seq in range(n_seq):
            train_data_seq = train_dataset.seqC_all[idx_seq, 1:]  # [14]
            train_data_chR = train_dataset.chR_all[idx_seq, idx_theta, :, :]  # [C, 1]
            train_data_seq = train_data_seq.repeat(C, 1)  # [C, 14]
            train_data[idx_seq] = torch.cat(
                [train_data_seq, train_data_chR], dim=1
            )  # [C, 15]

    return train_data, theta_value


idx_theta = 0
train_data, theta_value = get_data_for_theta(idx_theta, train_dataset)  # [MS, C, 15]
valid_data, theta_value = get_data_for_theta(idx_theta, valid_dataset)  # [MS, C, 15]


# run posterior and plot
fig_x, _ = plot_posterior_with_label(
    posterior=posterior,
    sample_num=config.train.posterior.sampling_num,
    x=seen_data.to(solver._device),  # TODO: change to data
    true_params=self.seen_data_for_posterior["theta"][fig_idx],
    limits=limits,
    prior_labels=prior_labels,
)

# generate posterior samples
cnle_samples = cnle_posterior.sample((num_samples,), x=x_o.reshape(num_trials, 2))

del solver
clean_cache()
print(f"PID: {PID} finished")


# if __name__ == "__main__":
#     main()
