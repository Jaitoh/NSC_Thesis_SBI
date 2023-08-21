import os
import sys
import torch
import h5py
import numpy as np
from sbi import analysis

from tqdm import tqdm
import pickle
from sklearn.manifold import TSNE
from scipy.spatial import distance
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import hydra

from pathlib import Path
import sys

NSC_DIR = Path(__file__).resolve().parent.parent.parent.parent.as_posix()  # NSC dir
sys.path.append(f"{NSC_DIR}/codes/src")

# from scipy.stats import gaussian_kde
from train.train_L0_p4a import Solver
from features.features import Feature_Generator
from simulator.model_sim_pR import DM_sim_for_seqCs_parallel_with_smaller_output
from utils.set_seed import setup_seed
from utils.inference import load_stored_config as load_config
from utils.inference import get_posterior, estimate_theta_from_post_samples, sampling_from_posterior
from utils.range import convert_array_range
from features.features import feature_extraction_fn

setup_seed(0)

plt.rcParams["font.size"] = 32
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 3

device = "cuda" if torch.cuda.is_available() else "cpu"

PID = os.getpid()
print(f"PID: {PID}")


class Feature_Gen_Dataset(Dataset):
    def __init__(self, seqC, chRs, FG, config):
        D, M, S = seqC.shape[0], seqC.shape[1], seqC.shape[2]
        self.seqC = seqC
        self.chRs = chRs
        self.D = D
        self.M = M
        self.S = S
        self.config = config
        self.FG = FG

    def __len__(self):
        return self.chRs.shape[-1]

    def __getitem__(self, idx):
        chR = self.chRs[..., idx]
        return feature_extraction_fn(self.seqC, chR, self.FG, self.config)


def extract_trained_features_from_seqC_chRs(seqC, chRs, config):
    """
    seqC: (D, M, S, 15)
    chR: (D, M, S, num_C)

    output features: (num_C, 1, num_features, 1)
    """
    FG = Feature_Generator()
    dataset = Feature_Gen_Dataset(seqC, chRs, FG, config)
    loader = DataLoader(dataset, num_workers=8, batch_size=1)

    return torch.cat(list(tqdm(loader)), dim=0)


# === compute one example distance ===
# load seqC, theta, probR
def load_data(DATA_PATH, solver):
    # train_set = solver.inference.train_set_names[0]
    # valid_set = solver.inference.valid_set_names[0]  # TODO:
    data_set = solver.inference.valid_set_names[0]  # TODO:

    # find the prior range limits
    prior_limits = solver._get_limits()

    # with h5py.File(DATA_PATH, "r") as dataset_file:
    dataset_file = h5py.File(DATA_PATH, "r")
    theta_idx = 0
    seqC = torch.from_numpy(dataset_file[data_set]["seqC"][:])
    D, M, S = seqC.shape[0], seqC.shape[1], seqC.shape[2]
    DMS = D * M * S
    theta = torch.from_numpy(dataset_file[data_set]["theta"][theta_idx, :]).view(1, -1)
    probR = torch.from_numpy(dataset_file[data_set]["probR"][..., theta_idx, :])
    print(f"==>> probR.shape: {probR.shape}")
    return prior_limits, seqC, D, M, S, theta, probR


def probR_to_chR(probR, num_C, device):
    "generate chR from sampling probR num_C times"
    if device == "cuda":
        probR = probR.cuda()
        probR = probR.repeat_interleave(num_C, dim=-1)
        chR = torch.bernoulli(probR).cpu()
    else:
        probR = probR.repeat_interleave(num_C, dim=-1)
        chR = torch.bernoulli(probR)
    return chR


# comput the feature from seqC and chR
# def compute_feature_from_seqC_chR(seqC, D, M, S, chR, chosen_features):
#     """
#     seqC: (D, M, S, 15)
#     chR: (D, M, S, num_C)

#     """
#     FG = Feature_Generator()
#     dataset = Feature_Gen_Dataset(seqC, chR, D, M, S, chosen_features, FG)
#     loader = DataLoader(dataset, num_workers=8, batch_size=1)
#     return torch.cat(list(tqdm(loader)), dim=0)


# def estimate_theta_from_post_samples(prior_limits, samples):
#     theta_estimated = []
#     for i in tqdm(range(len(prior_limits))):
#         kde = gaussian_kde(samples[:, i])
#         prior_range = np.linspace(prior_limits[i][0], prior_limits[i][1], 2500)
#         densities = kde.evaluate(prior_range)
#         theta_value = prior_range[np.argmax(densities)]
#         theta_estimated.append(theta_value)
#     return theta_estimated


# LOG_DIR = "/home/ubuntu/tmp/NSC/codes/src/train/logs"
# EXP_ID = "train_L0_p4/p4-4Fs-1D-cnn"
# DATA_PATH = "/home/ubuntu/tmp/NSC/data/dataset/dataset-L0-Eset0-100sets-T500.h5"
# num_C = 100
# # num_C = config.dataset.partial_C  # number of Ch samplings
# num_estimation = 3


@hydra.main(config_path="./", config_name="config", version_base=None)
def main(_config):
    LOG_DIR = _config.LOG_DIR
    OUT_DIR = _config.OUT_DIR
    EXP_ID = _config.EXP_ID
    DATA_PATH = _config.DATA_PATH
    num_C = _config.num_C
    # num_C = config.dataset.partial_C  # number of Ch samplings
    num_estimation = _config.num_estimation
    # create features and fig folders
    os.makedirs(f"{OUT_DIR}/features", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/fig", exist_ok=True)

    model_path, config = load_config(LOG_DIR, EXP_ID)
    chosen_features = config.dataset.concatenate_feature_types

    # get the trained posterior
    solver, posterior = get_posterior(model_path, config, device, Solver=Solver)

    # load seqC, theta, probR
    prior_limits, seqC, D, M, S, theta, probR = load_data(DATA_PATH, solver)

    # compute features_true
    chR = probR_to_chR(probR, num_C, device)
    features_true = compute_feature_from_seqC_chR(seqC, D, M, S, chR, chosen_features)
    print(f"==>> features_true.shape: {features_true.shape}")

    # pickle features_true
    with open(f"{OUT_DIR}/features/features_true.pkl", "wb") as f:
        pickle.dump(features_true, f)
    print(f"==>> features_true.pkl saved\n")

    feature_estimated_collection = []
    theta_estimated_collection = []
    for i in range(num_estimation):
        # sample from the posterior with x
        feature = features_true[i]
        samples = sampling_from_posterior(device, posterior, feature)
        # find estimated theta using "kde"
        theta_estimated = estimate_theta_from_post_samples(prior_limits, samples)

        # print theta with 2 decimal places
        theta_estimated = np.array(theta_estimated).reshape(1, -1)
        print(f"==>> theta: {theta}")
        print(f"==>> theta_estimated: {np.round(theta_estimated, 2)}")
        theta_estimated_collection.append(np.round(theta_estimated, 2))

        # compute the probR based on the estimated theta
        _, probR_estimated = DM_sim_for_seqCs_parallel_with_smaller_output(
            seqCs=seqC,
            prior=theta_estimated,
            num_prior_sample=1,
            privided_prior=True,
            model_name="B-G-L0S-O-N-",
        )
        probR_estimated = torch.from_numpy(probR_estimated.reshape(D, M, S, 1))
        chR_estimated = probR_to_chR(probR_estimated, num_C, device)

        # compute the feature from seqC and chR_estimated
        features_estimated = compute_feature_from_seqC_chR(seqC, D, M, S, chR_estimated, chosen_features)
        print(f"==>> features_estimated.shape: {features_estimated.shape}")
        feature_estimated_collection.append(features_estimated)

    # pickle feature_estimated_collection
    with open(
        f"{OUT_DIR}/features/feature_estimated_collection.pkl",
        "wb",
    ) as f:
        pickle.dump(feature_estimated_collection, f)
    print(f"==>> feature_estimated_collection.pkl saved")

    for i in range(num_estimation):
        # embedding to 2D
        tsne = TSNE(n_components=2, random_state=0, perplexity=5)
        reduced_features_true = tsne.fit_transform(features_true.squeeze().numpy())
        reduced_features_estimated = tsne.fit_transform(feature_estimated_collection[i].squeeze().numpy())

        # compute distances - inner distances
        distances_true = distance.pdist(reduced_features_true, "euclidean")
        distances_estimated = distance.pdist(reduced_features_estimated, "euclidean")
        distances_true_square = distance.squareform(distances_true)
        distances_estimated_square = distance.squareform(distances_estimated)
        distances_inter = distance.cdist(reduced_features_true, reduced_features_estimated, "euclidean")

        # Get upper triangular part without the diagonal (k=1)
        upper_part_true = distances_true_square[np.triu_indices(distances_true_square.shape[0], k=1)]
        upper_part_estimated = distances_estimated_square[
            np.triu_indices(distances_estimated_square.shape[0], k=1)
        ]

        # plot ==============================
        fig = plt.figure(figsize=(85, 30))
        # axs = axs.flatten()
        grid = plt.GridSpec(2, 5, wspace=0.2, hspace=0.5)
        ax0 = plt.subplot(grid[0:2, 0:2])
        ax01 = plt.subplot(grid[0, 2])
        ax02 = plt.subplot(grid[0, 3])
        ax03 = plt.subplot(grid[0, 4])
        ax11 = plt.subplot(grid[1, 2])
        ax12 = plt.subplot(grid[1, 3])
        ax13 = plt.subplot(grid[1, 4])

        fig_, axes = analysis.pairplot(
            samples,
            limits=prior_limits,
            # ticks=[[], []],
            figsize=(10, 10),
            points=theta.cpu().numpy(),
            points_offdiag={"markersize": 5, "markeredgewidth": 1},
            points_colors="r",
            labels=config.prior.prior_labels,
            upper=["kde"],
            diag=["kde"],
        )
        # save figure
        fig_.savefig("pairplot.png")
        fig_ = plt.imread("pairplot.png")
        ax0.imshow(fig_)
        title_name = f"Posterior estimation with validation data\nEXP: {EXP_ID}, {i}th choice samples\nTrue theta:{theta}\nestimated theta:{theta_estimated_collection[i]}"
        ax0.set_title(title_name)
        os.remove("pairplot.png")
        del fig_, axes

        im = ax01.imshow(distances_true_square)
        plt.colorbar(im)
        ax01.set_title("True embedding's\ndistances matrix")
        ax01.set_xlabel("True embedding idx")
        ax01.set_ylabel("True embedding idx")

        im = ax02.imshow(distances_estimated_square)
        plt.colorbar(im)
        ax02.set_title("Estimated embedding's\ndistances matrix")
        ax02.set_xlabel("Estimated embedding idx")
        ax02.set_ylabel("Estimated embedding idx")

        im = ax03.imshow(distances_inter)
        plt.colorbar(im)
        ax03.set_title("Inter embedding's\ndistances matrix")
        ax03.set_xlabel("True embedding idx")
        ax03.set_ylabel("Estimated embedding idx")

        # scatter plot
        ax11.scatter(
            reduced_features_true[:, 0], reduced_features_true[:, 1], s=20, color="b", label="True embedding"
        )
        ax11.scatter(
            reduced_features_estimated[:, 0],
            reduced_features_estimated[:, 1],
            s=20,
            color="r",
            label="Estimated embedding",
        )
        ax11.legend()
        ax11.set_title("t-SNE 100 feature embedding")

        # Plot histogram
        ax12.hist(upper_part_true, bins=30, alpha=0.5, label="True embedding's distances")
        ax12.hist(upper_part_estimated, bins=30, alpha=0.5, label="Estimated embedding's distances")
        ax12.set_title("Histogram of distances")
        ax12.set_xlabel("Distance")
        ax12.set_ylabel("Frequency")
        ax12.grid(alpha=0.25)
        ax12.legend()

        ax13.plot(features_true.squeeze()[0, :].numpy(), label="True")
        # ax13.plot(features_true.squeeze()[1, :].numpy())
        ax13.plot(feature_estimated_collection[i].squeeze()[0, :].numpy(), label="Estimated")
        # ax12.plot(feature_estimated_collection.squeeze()[1, :].numpy())
        ax13.set_title("Example of extracted feature")
        ax13.set_xlabel("feature index")
        ax13.set_ylabel("feature value")
        ax13.grid(alpha=0.25)
        ax13.legend()

        fig.tight_layout()
        EXP_ID = str(config.exp_id).split("/")[-1]
        fig_name = f"{OUT_DIR}/fig/{EXP_ID}-eval_{i}.png"
        fig.savefig(fig_name)


if __name__ == "__main__":
    main()
