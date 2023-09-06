"""inference with subject data

load posterior
load and process trial data
inference with trial data
"""
import torch
import pickle
import argparse
from pathlib import Path
import sys
import numpy as np

NSC_DIR = Path(__file__).resolve().parent.parent.parent.parent.as_posix()  # NSC dir
sys.path.append(f"{NSC_DIR}/codes/src")

from utils.subject import get_xo, parse_trial_data
from utils.inference import (
    load_stored_config,
    get_posterior,
    estimate_theta_from_post_samples,
    # convert_normed_theta,
    sampling_from_posterior,
)
from utils.train import plot_posterior_unseen, get_limits
from utils.setup import adapt_path
from features.features import feature_extraction
from utils.range import x2seqC, seqC2x, convert_samples_range


def estimate_theta_for_usr_npe(subj_ID, exp_dir, pipeline_version="p4a"):
    # load config
    config, model_path = load_stored_config(exp_dir=exp_dir)
    if "p4" in pipeline_version:
        from train.train_L0_p4a import Solver
    if "p5" in pipeline_version:
        from train.train_L0_p5a import Solver
    # load trained posterior model
    solver, posterior = get_posterior(
        model_path=model_path,
        config=config,
        device="cuda",
        Solver=Solver,
        low_batch=10,
        return_dataset=False,
    )

    normed_limits = solver._get_limits()
    designed_limits = get_limits(config.prior.prior_min, config.prior.prior_max)

    if "chosen_dur_list" not in config.dataset.keys():
        chosen_dur_list = [3, 5, 7, 9, 11, 13, 15]
    else:
        chosen_dur_list = config.dataset.chosen_dur_list

    data_path = Path(NSC_DIR) / "data/trials.mat"
    if "p4" in pipeline_version:
        # === load trial data
        # seqC [D, M, S, 15] [-1, 1] nan
        # chR [D, M, S, 1]
        D = len(chosen_dur_list)
        MS_list = [0.2, 0.4, 0.8]

        data_subjects = parse_trial_data(data_path)
        seqC_o = np.zeros((D, 3, 700, 15))
        chR_o = np.zeros((D, 3, 700, 1))

        seqCs = data_subjects["pulse"]
        chRs = data_subjects["chooseR"]

        # check element in durs in dur_list, of subject and return the index
        subj_IDs = data_subjects["subjID"]
        idx_subj = subj_IDs == subj_ID
        durs = data_subjects["dur"]
        MSs = data_subjects["MS"]

        for i_D, dur in enumerate(chosen_dur_list):
            for i_MS, MS in enumerate(MS_list):
                idx_dur = np.isin(durs, dur)
                idx_MS = np.isin(MSs, MS)
                idx_chosen = idx_dur & idx_subj & idx_MS
                idx_chosen = idx_chosen.reshape(-1)
                seqC_o[i_D, i_MS, :, :] = seqCs[idx_chosen, :]
                chR_o[i_D, i_MS, :, :] = chRs[idx_chosen, :]
        seqC_o, chR_o = torch.from_numpy(seqC_o), torch.from_numpy(chR_o)

        # === extract feature
        F_o = feature_extraction(seqC_o, chR_o, config).unsqueeze(0).unsqueeze(-1).cuda()
        print(f"==>> F_o.shape: {F_o.shape}")
        samples = sampling_from_posterior(
            "cuda",
            posterior,
            F_o,
            num_samples=20_000,
            show_progress_bars=True,
        )
        samples = convert_samples_range(samples, normed_limits, designed_limits)
        dest_limits = designed_limits
        theta_estimated = estimate_theta_from_post_samples(dest_limits, samples, mode="mode")

    if "p5" in pipeline_version:
        x_o, chR = get_xo(
            data_path,
            subj_ID=subj_ID,
            dur_list=chosen_dur_list,
            MS_list=[0.2, 0.4, 0.8],
        )
        # x_o [6300, 15], [0~1]
        # chR [6300, 1]

        xy_o = torch.cat([x_o, chR], dim=-1)

        samples = sampling_from_posterior(
            "cuda",
            posterior,
            xy_o,
            num_samples=20_000,
            show_progress_bars=False,
        )
        samples = convert_samples_range(samples, normed_limits, designed_limits)
        dest_limits = designed_limits
        theta_estimated = estimate_theta_from_post_samples(dest_limits, samples, mode="mode")

    print("\n" + "-" * 20 + f" theta_estimated for subject {subj_ID} " + "-" * 20)
    for label, estimate, range_ in zip(config.prior.prior_labels, theta_estimated, designed_limits):
        print(f"{label:10}: {estimate:8.3f} from range [{range_[0]:.3f}, {range_[1]:.3f}]")

    return theta_estimated, designed_limits, config.prior.prior_labels, samples


if __name__ == "__main__":
    pipeline_version = "p4a"
    exp_dir = "~/data/NSC/codes/src/train/logs/train_L0_p4a/p4a-F1345-cnn-maf3"

    # pipeline_version = "p5a"
    # exp_dir = "~/data/NSC/codes/src/train/logs/train_L0_p5a/p5a-conv_lstm-corr_conv"

    # exp_dir = "~/tmp/NSC/codes/src/train/logs/train_L0_p5a/p5a-conv_net"
    exp_dir = adapt_path(exp_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_dir", type=str, default=exp_dir, help="log directory")
    parser.add_argument(
        "-p", "--pipeline_version", type=str, default=pipeline_version, help="pipeline version"
    )
    args = parser.parse_args()

    exp_dir = adapt_path(args.exp_dir)
    pipeline_version = args.pipeline_version

    subj_thetas = {}
    subj_thetas["exp"] = str(exp_dir).split("/")[-2:]
    for subj_ID in range(2, 13):
        theta_estimated, range_, labels, samples = estimate_theta_for_usr_npe(
            subj_ID,
            exp_dir,
            pipeline_version,
        )
        subj_thetas["range_"] = range_
        subj_thetas["labels"] = labels
        subj_thetas[subj_ID] = theta_estimated

        # save samples
        with open(f"{exp_dir}/inference/samples_{subj_ID}.pkl", "wb") as f:
            pickle.dump(samples, f)

    Path(f"{exp_dir}/inference").mkdir(parents=True, exist_ok=True)
    # save pkl
    with open(f"{exp_dir}/inference/subj_thetas.pkl", "wb") as f:
        pickle.dump(subj_thetas, f)
    # save txt
    with open(f"{exp_dir}/inference/subj_thetas.txt", "w") as f:
        for subj_ID in range(2, 13):
            f.write(f"subj_ID={subj_ID}\n")
            for label, estimate in zip(labels, subj_thetas[subj_ID]):
                f.write(f"{label:10}: {estimate:8.3f}\n")
            f.write("\n")
