from pathlib import Path
from omegaconf import OmegaConf
import sys
import torch
import h5py
import numpy as np
from sbi import analysis
from scipy.stats import gaussian_kde
from tqdm import tqdm

sys.path.append("./src")
sys.path.append("../../src")

from train.train_L0_p4 import Solver
from features.features import Feature_Generator
from simulator.model_sim_pR import DM_sim_for_seqCs_parallel

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
solver.init_inference().prepare_dataset_network(config, model_path, device="cpu")

posterior = solver.inference.build_posterior(solver.inference._neural_net)
solver.inference._model_bank = []


# load seqC, theta, probR
train_set = solver.inference.train_set_names[0]
# valid_set = solver.inference.valid_set_names[0]

# with h5py.File(DATA_PATH, "r") as dataset_file:
dataset_file = h5py.File(DATA_PATH, "r")
theta_idx = 10
seqC = torch.from_numpy(dataset_file[train_set]["seqC"][:])
D, M, S = seqC.shape[0], seqC.shape[1], seqC.shape[2]
DMS = D * M * S
theta = torch.from_numpy(dataset_file[train_set]["theta"][theta_idx, :])
probR = torch.from_numpy(dataset_file[train_set]["probR"][..., theta_idx, :])

# comput the feature from seqC and chR
# for i in range(1):
chR = torch.bernoulli(probR)
chosen_features = config.dataset.concatenate_feature_types
FG = Feature_Generator()
feature = (
    FG.compute_kernels(seqC, chR, D, M, S)
    .get_provided_feature(chosen_features)
    .view(1, -1, 1)
)
print(f"==>> feature.shape: {feature.shape}")

# sample from the posterior with x
samples = posterior.sample((20000,), x=feature, show_progress_bars=True)

# plot the posterior
prior_limits = solver._get_limits()

# find estimated theta using "kde"
theta_estimated = []
for i in tqdm(range(len(prior_limits))):
    kde = gaussian_kde(samples[:, i])
    prior_range = np.linspace(prior_limits[i][0], prior_limits[i][1], 5000)
    densities = kde.evaluate(prior_range)
    theta_value = prior_range[np.argmax(densities)]
    theta_estimated.append(theta_value)

# print theta with 2 decimal places
theta_estimated = np.array(theta_estimated)
print(f"==>> theta_estimated: {np.round(theta_estimated, 2)}")

# compute the probR based on the estimated theta
seqC, theta, probR = DM_sim_for_seqCs_parallel(
    seqCs=seqC,
    prior=theta_estimated,
    privided_prior=True,
    num_prior_sample=0,
    model_name="B-G-L0S-O-N-",
)
