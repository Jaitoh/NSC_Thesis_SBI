"""check posterior for the trained model"""
import os
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize
from omegaconf import OmegaConf
import scipy.io as sio
import numpy as np
import sys
from pathlib import Path

NSC_DIR = Path(__file__).resolve().parent.parent.parent.parent.as_posix()  # NSC dir
sys.path.append(f"{NSC_DIR}/codes/src")

from train_nle.train_p2 import Solver

# from train_nle.MyLikelihoodEstimator import MyLikelihoodEstimator
from utils.setup import check_path, clean_cache, adapt_path
from utils.set_seed import setup_seed
from utils.train import (
    WarmupScheduler,
    plot_posterior_with_label,
    plot_posterior_unseen,
    load_net,
)
from utils.subject import get_xo
from utils.dataset.dataset import pR2cR_acc


def get_params(config):
    p_seq = config.posterior.p_seq
    n_chR = config.posterior.n_chR
    idx_theta = config.posterior.idx_theta
    return p_seq, n_chR, idx_theta


# prepare data for posterior
def get_data_for_theta_from_dataset(
    idx_theta,
    dataset,
    p_seq=1,
    n_chR=10,
):
    """
    get the data from the dataset for a given theta
    """
    # + dataset.seqC_all.shape  # [MS, 15]
    # + dataset.probR_all.shape  # [MS, T, 1]
    # + dataset.theta_all.shape  # [T, 4]

    theta_value = dataset.theta_all[idx_theta]

    probR = dataset.probR_all[:, idx_theta]  # [MS, 1]
    chR = pR2cR_acc(probR, n_chR)  # [MS, n_chR, 1]

    seqC = dataset.seqC_all[:, 1:]  # [MS, 14]
    seqC = seqC[:, None, :].repeat_interleave(n_chR, dim=1)  # [MS, n_chR, 14]

    seqC_chR = torch.cat([seqC, chR], dim=-1)  # [MS, n_chR, 15]

    # randomly choose n_seq sequences from MS sequences
    n_seq = int(p_seq * dataset.M * dataset.S)
    n_seq = 1 if n_seq == 0 else n_seq
    idx_seq = np.random.choice(dataset.M * dataset.S, n_seq, replace=False)

    seqC_chR = seqC_chR[idx_seq]  # [n_seq, n_chR, 15]

    # + seqC_chR: [n_seq, n_chR, 15]
    # + theta_value: [4]
    return seqC_chR, theta_value


def get_observed_data(
    config,
    solver,
    train_dataset,
    valid_dataset,
    from_dataset,
):
    if from_dataset == "train":
        p_seq, n_chR, idx_theta = get_params(config)

        # get simulated data for a given theta idx_theta
        # data [MS, C, 15], theta_value [4]
        train_data, theta_value = get_data_for_theta_from_dataset(
            idx_theta,
            train_dataset,
            p_seq=p_seq,
            n_chR=n_chR,
        )
        x_obs = train_data.reshape(-1, train_data.shape[-1]).to(solver.inference._device)

        fig_name = f"posterior/obs_Train_theta{idx_theta}_pseq{p_seq}_nchR_{n_chR}.png"

        print("".center(50, "-"), "\n")
        print(f"==>> theta_value: {theta_value}")
        print(f"==>> (obs)train_data.shape: {x_obs.shape}")
        print(f"==>> fig_name: {fig_name}")
        print("\n", "".center(50, "-"))

    elif from_dataset == "valid":
        p_seq, n_chR, idx_theta = get_params(config)

        # get simulated data for a given theta idx_theta
        # data [MS, C, 15], theta_value [4]
        valid_data, theta_value = get_data_for_theta_from_dataset(
            idx_theta,
            valid_dataset,
            p_seq=p_seq,
            n_chR=n_chR,
        )
        x_obs = valid_data.reshape(-1, valid_data.shape[-1]).to(solver.inference._device)

        fig_name = f"posterior/obs_Valid_theta{idx_theta}_pseq{p_seq}_nchR_{n_chR}.png"

        print("".center(50, "-"), "\n")
        print(f"==>> theta_value: {theta_value}")
        print(f"==>> (obs)valid_data.shape: {x_obs.shape}")
        print(f"==>> fig_name: {fig_name}")
        print("\n", "".center(50, "-"))

    elif from_dataset.startswith("s"):
        # load subject data
        data_path = Path(config.data_path)
        trial_path = data_path / "../trials.mat"
        trial_path = adapt_path(trial_path)
        subj_id = int(from_dataset[1:])
        # load subject data
        seqC, chR = get_xo(
            trial_path,
            subj_ID=subj_id,
            dur_list=config.dataset.chosen_dur_list,
            MS_list=config.experiment_settings.chosen_MS_list,
        )

        x_obs = torch.cat([seqC, chR], dim=1).to(solver.inference._device)
        fig_name = f"posterior/obs_Subject{subj_id}.png"
        print(f"==>> fig_name: {fig_name}")

        theta_value = None

    return x_obs, theta_value, fig_name


def get_dataset(
    config,
    model_path,
):
    solver = Solver(config, store_config=False)
    solver.init_inference(
        iid_batch_size_x=config.posterior.MCMC_iid_batch_size_x,  #!
        iid_batch_size_theta=config.posterior.MCMC_iid_batch_size_theta,  # + info: 10000 MCMC init, other time 1
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
        print_info=True,
        inference_mode=True,
    )

    return solver, density_estimator, train_dataset, valid_dataset


def get_posterior(config):
    model_path = adapt_path(f"{config.log_dir}/model/model_check_point.pt")

    log_dir = adapt_path(config.log_dir)
    # data_path = Path(config.data_path)
    setup_seed(config.seed)

    # ========== initialize solver and network ==========
    solver, density_estimator, train_dataset, valid_dataset = get_dataset(
        config,
        model_path,
    )

    # ========== get observed data ==========
    from_dataset = config.posterior.xo_dataset_name
    x_obs, theta_value, fig_name = get_observed_data(
        config,
        solver,
        train_dataset,
        valid_dataset,
        from_dataset,
    )

    # ========== build MCMC posterior ==========
    p_seq, n_chR, idx_theta = get_params(config)
    mcmc_parameters = dict(
        warmup_steps=config.posterior.warmup_steps,  #!
        thin=config.posterior.thin,  #!
        num_chains=min(os.cpu_count() - 1, config.posterior.num_chains),  #!
        # num_chains=1,
        num_workers=config.posterior.num_workers,  #!
        # num_workers=1,  #! for test
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

    if from_dataset == "train" or from_dataset == "valid":
        # run posterior and plot
        fig_x, _, samples = plot_posterior_with_label(
            posterior=posterior,
            sample_num=config.posterior.sampling_num,
            x=x_obs,
            true_params=theta_value,
            limits=solver._get_limits(),
            prior_labels=config.prior.prior_labels,
            show_progress_bars=True,
        )
        sample_name = f"posterior/samples_obs_{from_dataset}_theta{idx_theta}_pseq{p_seq}_nchR_{n_chR}.pt"

    elif from_dataset.startswith("s"):
        fig_x, _, samples = plot_posterior_unseen(
            posterior=posterior,
            sample_num=config.posterior.sampling_num,
            x=x_obs,
            limits=solver._get_limits(),
            prior_labels=config.prior.prior_labels,
            show_progress_bars=True,
        )
        subj_id = int(from_dataset[1:])
        sample_name = f"posterior/samples_obs_Subject{subj_id}.pt"

    # save posterior samples
    samples = samples.cpu()
    torch.save(samples, adapt_path(log_dir / sample_name))
    # np.save(log_dir / sample_name, samples)

    # save figure
    fig_x.savefig(  # save figure
        adapt_path(log_dir / fig_name),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    print(f"==>> saved figure: {log_dir / fig_name}")
    del solver


@hydra.main(config_path="../config_nle", config_name="config-nle-p2", version_base=None)
def main(config: DictConfig):
    # hydra.core.global_hydra.GlobalHydra.instance().clear()
    # initialize(config_path="../config_nle", job_name="test_nle")
    # config = compose(config_name="config-nle-test-t4")
    # print(OmegaConf.to_yaml(config))

    PID = os.getpid()
    print(f"PID: {PID}")

    get_posterior(config=config)

    clean_cache()
    print(f"PID: {PID} finished")


if __name__ == "__main__":
    main()
