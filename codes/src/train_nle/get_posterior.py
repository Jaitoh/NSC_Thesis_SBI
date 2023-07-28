"""check posterior for the trained model"""
import os
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize
from omegaconf import OmegaConf
import scipy.io as sio

import sys
from pathlib import Path

NSC_DIR = Path(__file__).resolve().parent.parent.parent.parent.as_posix()  # NSC dir
sys.path.append(f"{NSC_DIR}/codes/src")

from train_nle.train import Solver

# from train_nle.MyLikelihoodEstimator import MyLikelihoodEstimator
from utils.setup import check_path, clean_cache
from utils.train import WarmupScheduler, plot_posterior_with_label, load_net
from parse_data.parse_trial_data import get_xo


# prepare data for posterior
def get_data_for_theta(idx_theta, dataset, chR=True):
    """
    get the data from the dataset for a given theta
    """
    theta_value = dataset.theta_all[idx_theta]

    if chR:
        # + dataset.seqC_all.shape  # [MS, 15]
        # + dataset.chR_all.shape  # [MS, T, C, 1]
        # + dataset.theta_all.shape  # [T, 4]
        n_seq = dataset.seqC_all.shape[0]
        T, C = dataset.chR_all.shape[1:3]
        data = torch.empty((n_seq, C, 15))  # [MS, C, 15]
        for idx_seq in range(n_seq):
            data_seq = dataset.seqC_all[idx_seq, 1:]  # [14]
            data_chR = dataset.chR_all[idx_seq, idx_theta, :, :]  # [C, 1]
            data_seq = data_seq.repeat(C, 1)  # [C, 14]
            data[idx_seq] = torch.cat([data_seq, data_chR], dim=1)  # [C, 15]
    # + data: [n_seq, C, 15]
    # + theta_value: [4]
    return data, theta_value


def get_posterior(idx_theta, config):
    model_path = f"{config.log_dir}/model/model_check_point.pt"

    log_dir = Path(config.log_dir)
    # data_path = Path(config.data_path)

    # initialize solver and network
    solver = Solver(config, store_config=False)
    solver.init_inference(
        iid_batch_size_x=300,  #!
        iid_batch_size_theta=-1,  #!
        sum_writer=False,
    )

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

    # get simulated data for a given theta
    # data [MS, C, 15], theta_value [4]
    train_data, theta_value = get_data_for_theta(idx_theta, train_dataset)
    valid_data, theta_value = get_data_for_theta(idx_theta, valid_dataset)

    print("".center(50, "-"), "\n")
    print(f"==>> theta_value: {theta_value}")
    print(f"==>> train_data.shape: {train_data.shape}")
    print(f"==>> valid_data.shape: {valid_data.shape}")
    print("\n", "".center(50, "-"))

    # ========== get observed data ==========
    from_dataset = config.posterior.xo_dataset_name
    if from_dataset == "train":
        n_seq = config.posterior.n_seq
        n_chR = config.posterior.n_chR
        x_obs = (
            train_data[:n_seq][:n_chR, :].reshape(-1, 15).to(solver.inference._device)
        )
        fig_name = f"posterior_theta{idx_theta}_obs_Train_seq{n_seq}_chR_{n_chR}.png"
        print(f"==>> fig_name: {fig_name}")

    elif from_dataset == "valid":
        n_seq = config.posterior.n_seq
        n_chR = config.posterior.n_chR
        x_obs = (
            valid_data[:n_seq][:n_chR, :].reshape(-1, 15).to(solver.inference._device)
        )
        fig_name = f"posterior_theta{idx_theta}_obs_Valid_seq{n_seq}_chR_{n_chR}.png"
        print(f"==>> fig_name: {fig_name}")

    elif from_dataset.startswith("s"):
        # load subject data
        data_path = Path(config.data_path)
        trial_path = data_path / "../trials.mat"
        trial_path = trial_path.expanduser()
        subj_id = int(from_dataset[1:])
        # load subject data
        seqC, chR = get_xo(
            trial_path,
            subj_ID=subj_id,
            dur_list=config.dataset.chosen_dur_list,
            MS_list=config.experiment_settings.chosen_MS_list,
        )

        x_obs = torch.cat([seqC, chR], dim=1).to(solver.inference._device)
        fig_name = f"posterior_theta{idx_theta}_obs_Subject{subj_id}.png"
        print(f"==>> fig_name: {fig_name}")

    # ========== build MCMC posterior ==========
    mcmc_parameters = dict(
        warmup_steps=100,  #!
        thin=1,  #!
        num_chains=min(os.cpu_count() - 1, 7),  #!
        # num_chains=1,
        num_workers=1,  #!
        init_strategy="sir",  #!
    )
    print(f"==>> mcmc_parameters: {mcmc_parameters}")

    posterior = solver.inference.build_posterior(
        density_estimator=density_estimator,
        prior=solver.inference._prior,
        sample_with="mcmc",  #!
        mcmc_method="slice",  #!
        # mcmc_method="slice_np",
        # mcmc_method="slice_np_vectorized",
        mcmc_parameters=mcmc_parameters,  #!
        # sample_with="vi",
        # vi_method="rKL",
        # vi_parameters={},
        # rejection_sampling_parameters={},
    )

    # run posterior and plot
    fig_x, _ = plot_posterior_with_label(
        posterior=posterior,
        sample_num=config.posterior.sampling_num,
        x=x_obs,
        true_params=theta_value,
        limits=solver._get_limits(),
        prior_labels=config.prior.prior_labels,
        show_progress_bars=True,
    )

    # save figure
    fig_x.savefig(  # save figure
        log_dir / fig_name,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    print(f"==>> saved figure: {log_dir / fig_name}")
    del solver


@hydra.main(
    config_path="../config_nle", config_name="config-nle-snn", version_base=None
)
def main(config: DictConfig):
    # hydra.core.global_hydra.GlobalHydra.instance().clear()
    # initialize(config_path="../config_nle", job_name="test_nle")
    # config = compose(config_name="config-nle-test-t4")
    # print(OmegaConf.to_yaml(config))

    PID = os.getpid()
    print(f"PID: {PID}")

    for idx_theta in range(1):
        print(f"==>> idx_theta: {idx_theta}")
        get_posterior(idx_theta=idx_theta, config=config)

    clean_cache()
    print(f"PID: {PID} finished")


if __name__ == "__main__":
    main()
