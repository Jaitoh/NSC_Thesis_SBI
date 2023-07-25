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
initialize(config_path="../config_nle", job_name="test_nle")
config = compose(config_name="config-nle-test-t4")
print(OmegaConf.to_yaml(config))

model_path = (
    Path(NSC_DIR)
    / "codes/src/train_nle/logs/L0-nle-cnn/L0-nle-cnn-dur3-online-copy/model/model_check_point.pt"
)


PID = os.getpid()
print(f"PID: {PID}")

log_dir = Path(config.log_dir)
data_path = Path(config.data_path)

# initialize solver and network
solver = Solver(config, store_config=False)
solver.init_inference()

# get the training dataset, & trained network
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


# prepare data for posterior
def get_data_for_theta(idx_theta, dataset, chR=True):
    """
    get the data from the dataset for a given theta
    """
    theta_value = dataset.theta_all[idx_theta]

    if chR:
        # dataset.seqC_all.shape  # [MS, 15]
        # dataset.chR_all.shape  # [MS, T, C, 1]
        # dataset.theta_all.shape  # [T, 4]
        n_seq = dataset.seqC_all.shape[0]
        T, C = dataset.chR_all.shape[1:3]
        data = torch.empty((n_seq, C, 15))  # [MS, C, 15]
        for idx_seq in range(n_seq):
            data_seq = dataset.seqC_all[idx_seq, 1:]  # [14]
            data_chR = dataset.chR_all[idx_seq, idx_theta, :, :]  # [C, 1]
            data_seq = data_seq.repeat(C, 1)  # [C, 14]
            data[idx_seq] = torch.cat([data_seq, data_chR], dim=1)  # [C, 15]

    return data, theta_value


idx_theta = 0
train_data, theta_value = get_data_for_theta(idx_theta, train_dataset)  # [MS, C, 15]
valid_data, theta_value = get_data_for_theta(idx_theta, valid_dataset)  # [MS, C, 15]

print("".center(50, "-"), "\n")
print(f"==>> theta_value: {theta_value}")
print(f"==>> train_data.shape: {train_data.shape}")
print(f"==>> valid_data.shape: {valid_data.shape}")
print("\n", "".center(50, "-"))


# build MCMC posterior
mcmc_parameters = dict(
    warmup_steps=100,
    thin=10,
    num_chains=1,
    num_workers=1,
    init_strategy="sir",
)

posterior = solver.inference.build_posterior(
    density_estimator=density_estimator,
    prior=solver.inference._prior,
    sample_with="mcmc",
    mcmc_method="slice_np",
    mcmc_parameters=mcmc_parameters,
    # vi_method="rKL",
    # vi_parameters={},
    # rejection_sampling_parameters={},
)


# samples = posterior.sample(
#     (2000,),
#     x=train_data[0].reshape(-1, 15).to("cpu"),
#     show_progress_bars=True,
# )

# run posterior and plot
print(f"==>> train_data[0].shape: {train_data[0][0:2,:].reshape(-1, 15).shape}")
fig_x, _ = plot_posterior_with_label(
    posterior=posterior,
    sample_num=config.posterior.sampling_num,
    x=train_data[0].reshape(-1, 15).to(solver.inference._device),
    true_params=theta_value,
    limits=solver._get_limits(),
    prior_labels=config.prior.prior_labels,
)


del solver
clean_cache()
print(f"PID: {PID} finished")
