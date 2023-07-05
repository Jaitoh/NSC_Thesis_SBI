from pathlib import Path
from omegaconf import OmegaConf
import os
import sys
import torch
import h5py
import numpy as np
from sbi import analysis
from scipy.stats import gaussian_kde
from tqdm import tqdm
import pickle
from sklearn.manifold import TSNE
from scipy.spatial import distance
import matplotlib.pyplot as plt

sys.path.append("./src")
sys.path.append("../../src")

from train.train_L0_p4 import Solver
from features.features import Feature_Generator
from simulator.model_sim_pR import DM_sim_for_seqCs_parallel_with_smaller_output


device = "cuda" if torch.cuda.is_available() else "cpu"

PID = os.getpid()
print(f"PID: {PID}")

# compute the distance between the predicted and the ground truth
LOG_DIR = "/home/ubuntu/tmp/NSC/codes/src/train/logs"
EXP_ID = "train_L0_p4/p4-4Fs-1D-cnn"
DATA_PATH = "/home/ubuntu/tmp/NSC/data/dataset/dataset-L0-Eset0-100sets-T500.h5"
# === compute one example distance ===
# load config
config_path = Path(LOG_DIR) / EXP_ID / "config.yaml"
model_path = Path(LOG_DIR) / EXP_ID / "model" / "best_model.pt"
config = OmegaConf.load(config_path)
config.log_dir = str(Path(LOG_DIR) / EXP_ID)

# get the trained posterior
solver = Solver(config)
solver.init_inference().prepare_dataset_network(config, model_path, device=device)

posterior = solver.inference.build_posterior(solver.inference._neural_net)
solver.inference._model_bank = []


# load seqC, theta, probR
train_set = solver.inference.train_set_names[0]
# valid_set = solver.inference.valid_set_names[0] # TODO:

# find the prior range limits
prior_limits = solver._get_limits()

# with h5py.File(DATA_PATH, "r") as dataset_file:
dataset_file = h5py.File(DATA_PATH, "r")
theta_idx = 0
seqC = torch.from_numpy(dataset_file[train_set]["seqC"][:])
D, M, S = seqC.shape[0], seqC.shape[1], seqC.shape[2]
DMS = D * M * S
theta = torch.from_numpy(dataset_file[train_set]["theta"][theta_idx, :]).view(1, -1)
probR = torch.from_numpy(dataset_file[train_set]["probR"][..., theta_idx, :])
print(f"==>> probR.shape: {probR.shape}")


num_C = config.dataset.partial_C  # number of Ch samplings
num_estimation = 3
chosen_features = config.dataset.concatenate_feature_types


# comput the feature from seqC and chR
def probR_2_chR(probR, num_C, device):
    if device == "cuda":
        probR = probR.cuda()
        probR = probR.repeat_interleave(num_C, dim=-1)
        chR = torch.bernoulli(probR).cpu()
    else:
        probR = probR.repeat_interleave(num_C, dim=-1)
        chR = torch.bernoulli(probR)
    return chR


chR = probR_2_chR(probR, num_C, device)

FG = Feature_Generator()

feature_collection = []
for i in tqdm(range(num_C)):
    chR_ = chR[..., i]
    feature = (
        FG.compute_kernels(seqC, chR_, D, M, S)
        .get_provided_feature(chosen_features)
        .view(1, -1, 1)
    )
    feature_collection.append(feature)
feature_collection = torch.cat(feature_collection, dim=0)

# pickle feature_collection
with open(
    "/home/ubuntu/tmp/NSC/codes/src/eval/features/feature_collection.pkl", "wb"
) as f:
    pickle.dump(feature_collection, f)
print(f"==>> feature_collection.pkl saved\n")

feature_estimated_collection_different_theta = []

for j in range(num_estimation):
    # sample from the posterior with x
    feature = feature_collection[i]
    samples = (
        posterior.sample(
            (20000,),
            x=feature.cuda() if device == "cuda" else feature,
            show_progress_bars=True,
        )
        .cpu()
        .numpy()
    )

    # find estimated theta using "kde"
    theta_estimated = []
    for i in tqdm(range(len(prior_limits))):
        kde = gaussian_kde(samples[:, i])
        prior_range = np.linspace(prior_limits[i][0], prior_limits[i][1], 5000)
        densities = kde.evaluate(prior_range)
        theta_value = prior_range[np.argmax(densities)]
        theta_estimated.append(theta_value)

    # print theta with 2 decimal places
    theta_estimated = np.array(theta_estimated).reshape(1, -1)
    print(f"==>> theta: {theta}")
    print(f"==>> theta_estimated: {np.round(theta_estimated, 2)}")

    # compute the probR based on the estimated theta
    _, probR_estimated = DM_sim_for_seqCs_parallel_with_smaller_output(
        seqCs=seqC,
        prior=theta_estimated,
        num_prior_sample=1,
        privided_prior=True,
        model_name="B-G-L0S-O-N-",
    )
    probR_estimated = torch.from_numpy(probR_estimated.reshape(D, M, S, 1))

    feature_estimated_collection = []
    # compute the feature from seqC and chR_estimated
    chR_estimated = probR_2_chR(probR_estimated, num_C, device)
    for i in tqdm(range(num_C)):
        chR_estimated_ = chR_estimated[..., i]
        FG = Feature_Generator()
        feature_estimated = (
            FG.compute_kernels(seqC, chR_estimated_, D, M, S)
            .get_provided_feature(chosen_features)
            .view(1, -1, 1)
        )
        # print(f"==>> feature_estimated.shape: {feature_estimated.shape}")

        feature_estimated_collection.append(feature_estimated)
    feature_estimated_collection = torch.cat(feature_estimated_collection, dim=0)
    print(
        f"==>> feature_estimated_collection.shape: {feature_estimated_collection.shape}"
    )
    feature_estimated_collection_different_theta.append(feature_estimated_collection)

    # pickle feature_estimated_collection
    with open(
        f"/home/ubuntu/tmp/NSC/codes/src/eval/features/feature_estimated_collection_{j}.pkl",
        "wb",
    ) as f:
        pickle.dump(feature_estimated_collection, f)
    print(f"==>> feature_estimated_collection_{j}.pkl saved")

fig, axes = analysis.pairplot(
    samples.cpu().numpy(),
    limits=prior_limits,
    # ticks=[[], []],
    figsize=(10, 10),
    points=theta.cpu().numpy(),
    points_offdiag={"markersize": 5, "markeredgewidth": 1},
    points_colors="r",
    labels=[],
    upper=["kde"],
    diag=["kde"],
)

# load feature_collection
with open(
    "/home/ubuntu/tmp/NSC/codes/src/eval/features/feature_collection.pkl", "rb"
) as f:
    feature_collection = pickle.load(f)

estimate_idx = 99
with open(
    f"/home/ubuntu/tmp/NSC/codes/src/eval/features/feature_estimated_collection_{estimate_idx}.pkl",
    "rb",
) as f:
    feature_estimated_collection = pickle.load(f)
plt.figure()
plt.plot(feature_collection.squeeze()[0, :].numpy())
plt.plot(feature_collection.squeeze()[1, :].numpy())
plt.plot(feature_collection.squeeze()[2, :].numpy())
plt.show()

plt.figure()
plt.plot(feature_estimated_collection.squeeze()[0, :].numpy())
plt.plot(feature_estimated_collection.squeeze()[1, :].numpy())
plt.plot(feature_estimated_collection.squeeze()[2, :].numpy())
plt.show()

print(f"==>> feature_collection.shape: {feature_collection.shape}")
print(f"==>> feature_estimated_collection.shape: {feature_estimated_collection.shape}")

tsne = TSNE(n_components=2, random_state=0)
reduced_features = tsne.fit_transform(feature_collection.squeeze().numpy())
reduced_features_estimated = tsne.fit_transform(
    feature_estimated_collection.squeeze().numpy()
)
# plt.scatter(reduced_features[:, 0], reduced_features[:, 1], s=1)
plt.scatter(reduced_features_estimated[:, 0], reduced_features_estimated[:, 1], s=1)

distances = distance.pdist(reduced_features, "euclidean")
distances_estimated = distance.pdist(reduced_features_estimated, "euclidean")

distances_square = distance.squareform(distances)
distances_estimated_square = distance.squareform(distances_estimated)

plt.figure()
plt.imshow(distances_square)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(distances_estimated_square)
plt.colorbar()
plt.show()

distances_inter = distance.cdist(
    reduced_features, reduced_features_estimated, "euclidean"
)
# distances_inter_square = distance.squareform(distances_inter)
plt.figure()
plt.imshow(distances_inter)
plt.colorbar()
plt.show()


# Get upper triangular part without the diagonal (k=1)
upper_part = distances_square[np.triu_indices(distances_square.shape[0], k=1)]

# Plot histogram
plt.figure(figsize=(10, 7))
plt.hist(upper_part, bins=30, alpha=0.5, color="b", edgecolor="black")
plt.title("Histogram of distances")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
