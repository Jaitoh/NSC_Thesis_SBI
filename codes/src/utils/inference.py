import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
from scipy.stats import gaussian_kde
from torch.utils.data import Dataset, DataLoader
import torch

from pathlib import Path
import sys

NSC_DIR = Path(__file__).resolve().parent.parent.parent.parent.as_posix()  # NSC dir
sys.path.append(f"{NSC_DIR}/codes/src")

from utils.setup import adapt_path


# load config
def load_stored_config(exp_dir):
    config_path = Path(exp_dir) / "config.yaml"
    config_path = adapt_path(config_path)
    print(f"==>> config_path: {config_path}")
    config = OmegaConf.load(config_path)
    config.log_dir = str(exp_dir)

    model_path = Path(exp_dir) / "model" / "best_model.pt"
    # check if model exists
    if not model_path.exists():
        model_path = Path(exp_dir) / "model" / "model_check_point.pt"

    return config, model_path


def get_posterior(model_path, config, device, Solver, low_batch=0, return_dataset=False):
    """get the trained posterior"""

    solver = Solver(config, training_mode=False)
    train_loader, valid_loader, network, train_dataset, valid_dataset = solver.init_inference(
        sum_writer=False
    ).prepare_dataset_network(
        config,
        model_path,
        device=device,
        low_batch=low_batch,
    )
    posterior = solver.inference.build_posterior(network)
    solver.inference._model_bank = []
    print(f"finished building posterior")
    if return_dataset:
        return solver, posterior, train_loader, valid_loader, train_dataset, valid_dataset
    else:
        return solver, posterior


def estimate_theta_from_post_samples(prior_limits, samples, mode="mode"):
    # move samples to cpu if necessary
    # if samples are torch tensors
    if type(samples) == torch.Tensor:
        if samples.is_cuda:
            samples = samples.cpu().detach().numpy()

    # estimate theta values from samples using KDE
    if mode == "mode":
        # theta_estimated = []
        # for i in range(len(prior_limits)):
        #     kde = gaussian_kde(samples[:, i])
        #     prior_range = np.linspace(prior_limits[i][0], prior_limits[i][1], 2500)
        #     densities = kde.evaluate(prior_range)
        #     theta_value = prior_range[np.argmax(densities)]
        #     theta_estimated.append(theta_value)
        # theta_estimated = np.array(theta_estimated)

        num_params = samples.shape[1]
        prior_ranges = [np.linspace(lim[0], lim[1], 2500) for lim in prior_limits]

        # Pre-allocate array
        theta_estimated = np.empty(num_params)

        # Iterate over parameters and estimate mode using KDE
        for i, prior_range in enumerate(prior_ranges):
            kde = gaussian_kde(samples[:, i])
            densities = kde(prior_range)
            theta_estimated[i] = prior_range[np.argmax(densities)]

    # compute the mean as the prediction
    if mode == "mean":
        theta_estimated = np.mean(samples, axis=0)

    if mode == "median":
        theta_estimated = np.median(samples, axis=0)

    return theta_estimated


def sampling_from_posterior(device, posterior, x_o, num_samples=20000, show_progress_bars=False):
    return (
        posterior.sample(
            (num_samples,),
            x=x_o.cuda() if device == "cuda" else x_o,
            show_progress_bars=show_progress_bars,
        )
        .cpu()
        .numpy()
    )


# def convert_normed_theta(theta_estimated_normed, prior_min, prior_max, prior_labels):
#     # ! use utils.range functions instead
#     # convert theta from 0-1 to original range

#     original_range_pair = [[x, y] for x, y in zip(prior_min, prior_max)]

#     theta_estimated = []
#     for i in range(len(theta_estimated_normed)):
#         theta_estimated.append(
#             theta_estimated_normed[i] * (original_range_pair[i][1] - original_range_pair[i][0])
#             + original_range_pair[i][0]
#         )

#     return theta_estimated, original_range_pair


def ci_perf_on_dset(posterior, credible_intervals, dataset, num_params):
    ci_matrix = np.zeros((len(credible_intervals), num_params))

    for xy, theta in tqdm(dataset):
        samples = sampling_from_posterior(
            "cuda",
            posterior,
            xy,
            num_samples=2000,
            show_progress_bars=False,
        )
        # compute the ci
        for i, credible_interval in enumerate(credible_intervals):
            edge = (100 - credible_interval) / 2
            lower, upper = np.percentile(samples, [edge, 100 - edge], axis=0)
            inside_interval = (lower <= np.array(theta)) & (np.array(theta) <= upper)
            ci_matrix[i, :] = ci_matrix[i, :] + inside_interval.astype(int)

    ci_matrix /= len(dataset)

    return ci_matrix


def samples_on_dset(posterior, dataset):
    samples_collection = []
    theta_s = []

    for xy, theta in tqdm(dataset):
        samples = sampling_from_posterior(
            "cuda",
            posterior,
            xy,
            num_samples=2000,
            show_progress_bars=False,
        )
        samples_collection.append(samples)
        theta_s.append(theta)

    return samples_collection, theta_s


def perfs_on_dset(
    samples_collect,
    theta_s,
    credible_intervals,
    num_params,
    prior_limits,
    mode="ci",
):
    ci_matrix = np.zeros((len(credible_intervals), num_params))
    est_mean_s, est_mode_s, est_median_s = [], [], []
    mse_mean = np.zeros((num_params))
    mse_mode = np.zeros((num_params))
    mse_median = np.zeros((num_params))

    for samples, theta in tqdm(zip(samples_collect, theta_s), total=len(samples_collect)):
        # compute the ci
        if mode == "ci":
            for i, credible_interval in enumerate(credible_intervals):
                edge = (100 - credible_interval) / 2
                lower, upper = np.percentile(samples, [edge, 100 - edge], axis=0)
                inside_interval = (lower <= np.array(theta)) & (np.array(theta) <= upper)
                ci_matrix[i, :] = ci_matrix[i, :] + inside_interval.astype(int)

        # compute the mean
        if mode == "mean":
            est_mean = estimate_theta_from_post_samples(prior_limits, samples, mode="mean")
            est_mean_s.append(est_mean)
            mse_mean += (est_mean - np.array(theta)) ** 2

        # compute the mode
        if mode == "mode":
            est_mode = estimate_theta_from_post_samples(prior_limits, samples, mode="mode")
            est_mode_s.append(est_mode)
            mse_mode += (est_mode - np.array(theta)) ** 2

        # compute the median
        if mode == "median":
            est_median = estimate_theta_from_post_samples(prior_limits, samples, mode="median")
            est_median_s.append(est_median)
            mse_median += (est_median - np.array(theta)) ** 2

    if mode == "ci":
        ci_matrix /= len(samples_collect)
        return ci_matrix

    if mode == "mean":
        mse_mean /= len(samples_collect)
        return mse_mean, est_mean_s

    if mode == "mode":
        mse_mode /= len(samples_collect)
        return mse_mode, est_mode_s

    if mode == "median":
        mse_median /= len(samples_collect)
        return mse_median, est_median_s


def ci_perf_on_dset_old(posterior, credible_intervals, dataset, num_params):
    ci_matrix = np.zeros((len(credible_intervals), num_params))

    for i, credible_interval in enumerate(credible_intervals):
        edge = (100 - credible_interval) / 2
        print(f"credible interval: {credible_interval}%")

        for xy, theta in tqdm(dataset):
            samples = sampling_from_posterior(
                "cuda",
                posterior,
                xy,
                num_samples=2000,
                show_progress_bars=False,
            )
            lower, upper = np.percentile(samples, [edge, 100 - edge], axis=0)

            inside_interval = (lower <= np.array(theta)) & (np.array(theta) <= upper)
            ci_matrix[i, :] = ci_matrix[i, :] + inside_interval.astype(int)

    ci_matrix /= len(dataset)
    return ci_matrix


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def my_c2st(
    X_0,
    X_1,
    num_params=4,
    n_epochs=100,
):
    # Concatenate and label the datasets
    X = torch.cat((torch.tensor(X_1), torch.tensor(X_0)), dim=0)
    y = torch.tensor([0] * len(X_1) + [1] * len(X_0))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a DataLoader
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    # Define a simple neural network classifier
    class Classifier(nn.Module):
        def __init__(self):
            super(Classifier, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(num_params, 10), nn.ReLU(), nn.Linear(10, 2)  # 4 input features  # 2 classes
            )

        def forward(self, x):
            return self.fc(x)

    # Training the model
    model = Classifier()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(n_epochs)):  # You can choose the number of epochs
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X.float())
            loss = criterion(output, batch_y.long())
            loss.backward()
            optimizer.step()

    # Evaluate the model on the test set
    with torch.no_grad():
        test_output = model(X_test.float())
        predictions = torch.argmax(test_output, dim=1)
        accuracy = (predictions == y_test.long()).float().mean()

    # print(f"C2ST accuracy: {accuracy.item()}")
    return accuracy.item()


# class CredibleIntervalDataset(Dataset):
#     def __init__(self, credible_interval, dataset, posterior, num_samples=2000):
#         self.credible_interval = credible_interval
#         self.dataset = dataset
#         self.posterior = posterior
#         self.num_samples = num_samples
#         self.edge = (100 - credible_interval) / 2

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         xy, theta = self.dataset[idx]

#         samples = sampling_from_posterior(
#             "cuda",
#             self.posterior,
#             xy,
#             num_samples=self.num_samples,
#             show_progress_bars=False,
#         )

#         lower, upper = np.percentile(samples, [self.edge, 100 - self.edge], axis=0)

#         inside_interval = (lower <= np.array(theta)) & (np.array(theta) <= upper)

#         return inside_interval.astype(int)


# def ci_perf_on_dset_par(posterior, credible_intervals, dataset, num_params, num_workers=4):
#     ci_matrix = torch.zeros((len(credible_intervals), num_params))

#     for i, credible_interval in enumerate(credible_intervals):
#         print(f"==>> credible_interval: {credible_interval}")
#         dataset_par = CredibleIntervalDataset(credible_interval, dataset, posterior)
#         dataloader = DataLoader(dataset_par, batch_size=1, shuffle=False, num_workers=num_workers)

#         for inside_interval in tqdm(dataloader):
#             ci_matrix[i, :] = ci_matrix[i, :] + inside_interval[0]

#     ci_matrix /= len(dataset)
#     return ci_matrix
