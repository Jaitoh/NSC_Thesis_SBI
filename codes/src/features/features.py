"""
extracte features from / make summarization of 
seqC and cR

"""
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
from pathlib import Path
import os
import time
from tqdm import tqdm
import multiprocessing
import argparse
from joblib import Parallel, delayed
from torch.utils.data import Dataset, DataLoader
import cProfile

from pathlib import Path
import sys

NSC_DIR = Path(__file__).resolve().parent.parent.parent.parent.as_posix()  # NSC dir
sys.path.append(f"{NSC_DIR}/codes/src")
from utils.setup import adapt_path

# class Feature_Generator_Dataset(Dataset):
#     def __init__(self, C, probR, seqC, D, M, S, chosen_features):
#         self.C = C
#         self.probR = probR
#         self.seqC = seqC
#         self.D = D
#         self.M = M
#         self.S = S
#         self.chosen_features = chosen_features

#     def __len__(self):
#         return self.C

#     def __getitem__(self, idx):
#         chR = torch.bernoulli(self.probR)
#         FG = Feature_Generator()
#         feature = (
#             FG.compute_kernels(self.seqC, chR, self.D, self.M, self.S)
#             .get_provided_feature(self.chosen_features)
#             .view(1, -1, 1)
#         )
#         return feature


# dataset = Feature_Generator_Dataset(
#     num_embeddings, probR, seqC, D, M, S, chosen_features
# )
# dataloader = DataLoader(dataset, batch_size=64, num_workers=4)

# feature_collection = []
# for features in tqdm(dataloader):
#     feature_collection.append(features)


class Feature_Generator:
    def __init__(self):
        self.MD_list = torch.arange(-10, 11)

    def compute_kernels(self, seqC, chR, D, M, S):
        """
        seqC of shape [D, M, S, 15] or [DMS, 15]
        chR of shape [D, M, S, 1] or [DMS, 1]
        """
        # convert into torch if not
        # if not torch.is_tensor(seqC):
        #     seqC = torch.tensor(seqC)
        # if not torch.is_tensor(chR):
        #     chR = torch.tensor(chR)

        DMS = D * M * S
        seqC = seqC.view(DMS, -1)  # [DMS, 15]
        seqC = seqC.type(torch.float32)
        chR = chR.view(DMS, -1)  # [DMS, 1]
        chR = chR.type(torch.float32)
        self.D, self.M, self.S, self.DMS = D, M, S, DMS

        # ================== seqC properties ==================
        # compute Dur - duration of each sequence
        Dur = torch.sum(~torch.isnan(seqC), dim=-1, dtype=torch.int32)  # [DMS, 1]
        seqC = seqC.nan_to_num()
        # compute MS - motion strength
        MS = self._get_MS(seqC, DMS)  # [DMS, 1]
        # compute MD - final motion direction
        MD = torch.sum(torch.sign(seqC), axis=-1, dtype=torch.int32)  # [DMS, 1]
        # compute nSwitch
        nSwitch = self._compute_nSwitch(seqC, DMS).view(DMS, -1)  # [DMS, 1]

        Dur_list = torch.unique(Dur, sorted=True)
        MS_list = torch.unique(MS, sorted=True)
        MD_list = torch.unique(MD, sorted=True)
        if hasattr(self, "MD_list"):
            MD_list = self.MD_list

        # if MD_list is in self, then assign MD_list = self.MD_list else MD_list = MD_list
        self.Dur_list, self.MS_list, self.MD_list = Dur_list, MS_list, MD_list

        # ================== get kernels ==================
        # compute feature stats_MD, dist_MD, stats_NS, dist_NS - for feature 1&2, 4
        (
            self.dist_MD,
            self.dist_NS,
            self.stats_MD,
            self.stats_NS,
        ) = self._compute_kernel_MD_NS(chR, D, M, S, Dur, MS, MD, nSwitch, Dur_list, MS_list, MD_list)

        # compute stats_MD2, dist_MD2 - for feature 3
        self.dist_MD2, self.stats_MD2 = self._compute_kernel_MD2(chR, D, M, S, MS, MD, MS_list, MD_list)

        # compute stats_psy - for feature 5
        self.stats_psy = self._compute_kernel_psy(seqC, chR, D, M, Dur, MS, Dur_list, MS_list)

        return self

    def get_features(self):
        # generate a mask for MD
        mask_shape = (7, 21)
        ranges = [(-2, 2), (-4, 4), (-5, 5)]
        mask_MD = self._get_mask_MD(shape=mask_shape, ranges=ranges)
        for i in range(self.M):
            self.stats_MD[:, i, :][mask_MD] = torch.nan

        # feature 1&2
        feature_1s, feature_2s = [], []
        for i in range(self.M):
            f1, f2 = self._extract_f1_f2(self.stats_MD[:, i, :])
            feature_1s.append(f1)
            feature_2s.append(f2)

        feature_1s = torch.stack(feature_1s, dim=0)
        feature_2s = torch.stack(feature_2s, dim=0)

        # feature 3
        # feature_3s = []
        # for i in range(3):
        #     feature_3s.append(self._extract_f3(self.stats_MD2[i, :], ranges=[-7, 7]))
        feature_3s = [self._extract_f3(self.stats_MD2[i, :], ranges=[-7, 7]) for i in range(3)]

        # generate a mask for NS
        ranges = [(-2, 2), (-4, 4)]
        mask_NS = self._get_mask_NS(shape=mask_shape, ranges=ranges)
        for i in range(3):
            self.stats_NS[:, i, :][mask_NS] = torch.nan
        feature_3s = torch.stack(feature_3s, dim=0)

        # feature 4
        # feature_4s = []
        # for i in range(self.M):
        #     feature_4s.append(self._extract_f4(self.stats_NS[:, i, :]))
        feature_4s = [self._extract_f4(self.stats_NS[:, i, :]) for i in range(self.M)]
        feature_4s = torch.stack(feature_4s, dim=0)

        # feature 5
        # feature_5s = []
        # for i in range(self.M):
        #     feature_5s.append(self._extract_f5(self.stats_psy[:, i, :]))
        feature_5s = [self._extract_f5(self.stats_psy[:, i, :]) for i in range(self.M)]
        feature_5s = torch.stack(feature_5s, dim=0)

        return feature_1s, feature_2s, feature_3s, feature_4s, feature_5s

    def get_provided_feature(self, feature_list=[1, 2, 3, 4, 5]):
        """cat all features together"""
        feature_1s, feature_2s, feature_3s, feature_4s, feature_5s = self.get_features()

        # concatenate features
        objects = []
        if 1 in feature_list:
            objects.append(feature_1s)
        if 2 in feature_list:
            objects.append(feature_2s)
        if 3 in feature_list:
            objects.append(feature_3s)
        if 4 in feature_list:
            objects.append(feature_4s)
        if 5 in feature_list:
            objects.append(feature_5s)

        # features = []
        # for i in range(self.M):
        #     features.append(torch.cat([obj[i] for obj in objects], dim=0))
        features = [torch.cat([obj[i] for obj in objects], dim=0) for i in range(self.M)]

        feature = torch.cat([feature for feature in features], dim=0)
        return feature

    def _extract_f5(self, stats_psy):
        D, P = stats_psy.shape
        # f5 = []
        # for i in range(P):
        #     starting_idx = i // 2
        #     f5.append(stats_psy[starting_idx:, i])
        f5 = [stats_psy[i // 2 :, i] for i in range(P)]
        f5 = torch.cat(f5, dim=0)
        return f5

    def _extract_f4(self, stats_NS):
        D, MD = stats_NS.shape
        # f4 = []
        # for idx_D in range(D):
        #     idx_nan = torch.isnan(stats_NS[idx_D, :])
        #     f4.append(stats_NS[idx_D, :][~idx_nan])
        f4 = [stats_NS[idx_D, :][~torch.isnan(stats_NS[idx_D, :])] for idx_D in range(D)]
        f4 = torch.cat(f4, dim=0)

        # plt.plot(f4, ".-", label="f4")
        # plt.grid(alpha=0.2)
        return f4

    def _extract_f3(self, stats_MD2, ranges=[-7, 7]):
        idx_mid_col = len(stats_MD2) // 2
        lower, upper = idx_mid_col + ranges[0], idx_mid_col + ranges[1]
        stats_MD2[:lower] = torch.nan
        stats_MD2[upper + 1 :] = torch.nan

        idx_nan = torch.isnan(stats_MD2)

        return stats_MD2[~idx_nan]

    def _extract_f1_f2(self, stats_MD):
        """extract f1 and f2 from
        stats_MD: (D, MD)
        """
        D, MD = stats_MD.shape
        # extract f1
        # f1 = []
        # for idx_D in np.arange(D):
        #     # remove the nan values in stats_MD
        #     idx_D = D - idx_D - 1
        #     idx_nan = torch.isnan(stats_MD[idx_D, :])
        #     f1.append(stats_MD[idx_D, :][~idx_nan])
        f1 = [stats_MD[D - idx_D - 1, :][~torch.isnan(stats_MD[D - idx_D - 1, :])] for idx_D in np.arange(D)]
        f1 = torch.cat(f1, dim=0)

        # extract f2
        # f2 = []
        # for idx_MD in np.arange(MD):
        #     idx_nan = torch.isnan(stats_MD[:, idx_MD])
        #     f2.append(stats_MD[:, idx_MD][~idx_nan])
        f2 = [stats_MD[:, idx_MD][~torch.isnan(stats_MD[:, idx_MD])] for idx_MD in np.arange(MD)]
        f2 = torch.cat(f2, dim=0)

        # plt.plot(f1, '.-', label='f1')
        # plt.plot(f2, '.-', label='f2')
        # plt.legend()
        # plt.grid(alpha=0.2)

        return f1, f2

    def _get_mask_NS(self, shape=(7, 21), ranges=[(-2, 2), (-4, 4), (-5, 5)], show_mask=False):
        """generate a mask for the NS kernel
        select only the first two
        """
        mask = torch.ones(shape, dtype=torch.bool)
        idx_mid_col = mask.shape[1] // 2

        for i, mask_range in enumerate(ranges):
            lower, upper = idx_mid_col + mask_range[0], idx_mid_col + mask_range[1]
            mask[i, lower : upper + 1] = 0

        mask[:, idx_mid_col] = 1

        # if show_mask:
        #     plt.figure()
        #     plt.imshow(mask.T.numpy())
        #     # show values on the mask
        #     for i in range(mask.shape[0]):
        #         for j in range(mask.shape[1]):
        #             if not mask[i, j]:
        #                 plt.text(i, j, "0", ha="center", va="center", color="w")

        return mask

    def _get_mask_MD(self, shape=(7, 21), ranges=[(-2, 2), (-4, 4), (-5, 5)], show_mask=False):
        """generate a mask for the 2D"""
        mask = torch.ones(shape, dtype=torch.bool)
        idx_mid_col = mask.shape[1] // 2

        for i in range(mask.shape[0]):
            mask_range = ranges[-1] if i >= len(ranges) else ranges[i]
            lower, upper = idx_mid_col + mask_range[0], idx_mid_col + mask_range[1]
            mask[i, lower : upper + 1] = 0

        # if show_mask:
        #     plt.figure()
        #     plt.imshow(mask.T.numpy())
        #     # show values on the mask
        #     for i in range(mask.shape[0]):
        #         for j in range(mask.shape[1]):
        #             if not mask[i, j]:
        #                 plt.text(i, j, "0", ha="center", va="center", color="w")

        return mask

    def _compute_kernel_psy(self, seqC, chR, D, M, Dur, MS, Dur_list, MS_list):
        stats_psy_p = torch.zeros((D, M, 15 - 1), dtype=torch.float32)
        stats_psy_n = torch.zeros((D, M, 15 - 1), dtype=torch.float32)

        for idx_D in range(D):
            idx_current_D = (Dur == Dur_list[idx_D]).squeeze()

            for idx_M in range(M):
                idx_current_M = (MS == MS_list[idx_M]).squeeze()

                for idx_P in range(1, 15):  # pulse position
                    idx_current_pP = seqC[:, idx_P] > 0  # positive pulse position
                    idx_current_nP = seqC[:, idx_P] < 0  # negative pulse position

                    idx_f5_p = idx_current_D & idx_current_M & idx_current_pP
                    chR_chosen = chR[idx_f5_p, :]
                    stats_psy_p[idx_D, idx_M, idx_P - 1] = torch.mean(chR_chosen)

                    idx_f5_n = idx_current_D & idx_current_M & idx_current_nP
                    chR_chosen = chR[idx_f5_n, :]
                    stats_psy_n[idx_D, idx_M, idx_P - 1] = torch.mean(chR_chosen)

        stats_psy = stats_psy_p - stats_psy_n
        stats_psy.nan_to_num_(0)
        return stats_psy

    def _compute_kernel_MD2(self, chR, D, M, S, MS, MD, MS_list, MD_list):
        dist_MD2 = torch.zeros((M, len(MD_list)), dtype=torch.float32)  # for feature 3
        stats_MD2 = torch.zeros((M, len(MD_list)), dtype=torch.float32)

        for idx_M in range(M):
            idx_current_M = (MS == MS_list[idx_M]).squeeze()

            for idx_MD in range(len(MD_list)):
                idx_current_MD = (MD == MD_list[idx_MD]).squeeze()

                # feature 3
                idx_f3 = idx_current_M & idx_current_MD
                if torch.sum(idx_f3) != 0:
                    chR_chosen = chR[idx_f3, :]
                    stats_MD2[idx_M, idx_MD] = torch.mean(chR_chosen)
                    dist_MD2[idx_M, idx_MD] = torch.sum(chR_chosen) / (D * S)

        return dist_MD2, stats_MD2

    def _compute_kernel_MD_NS(self, chR, D, M, S, Dur, MS, MD, nSwitch, Dur_list, MS_list, MD_list):
        dist_MD = torch.zeros((D, M, len(MD_list)), dtype=torch.float32)  # for feature 1&2
        dist_NS = torch.zeros((D, M, len(MD_list)), dtype=torch.float32)  # for feature 4
        stats_MD = torch.zeros((D, M, len(MD_list)), dtype=torch.float32)
        stats_NS = torch.zeros((D, M, len(MD_list)), dtype=torch.float32)

        for idx_D in range(D):
            idx_current_D = (Dur == Dur_list[idx_D]).squeeze()

            for idx_M in range(M):
                idx_current_M = (MS == MS_list[idx_M]).squeeze()

                for idx_MD in range(len(MD_list)):
                    # idx_D, idx_M, idx_MD = 0, 0, 9
                    idx_current_MD = (MD == MD_list[idx_MD]).squeeze()
                    idx_current_NS = (nSwitch == 0).squeeze()

                    # feature 1&2
                    idx_f12 = idx_current_D & idx_current_M & idx_current_MD
                    if torch.sum(idx_f12) != 0:
                        # compute the stats for the current MD
                        chR_chosen = chR[idx_f12, :]
                        stats_MD[idx_D, idx_M, idx_MD] = torch.mean(chR_chosen)
                        dist_MD[idx_D, idx_M, idx_MD] = torch.sum(idx_f12) / S

                    # feature 4
                    idx_f4 = idx_current_D & idx_current_M & idx_current_MD & idx_current_NS
                    if torch.sum(idx_f4) != 0:
                        chR_chosen = chR[idx_f4, :]
                        stats_NS[idx_D, idx_M, idx_MD] = torch.mean(chR_chosen)
                        dist_NS[idx_D, idx_M, idx_MD] = torch.sum(chR_chosen) / S

        return dist_MD, dist_NS, stats_MD, stats_NS

    def _get_MS(self, seqC, DMS):
        # # compute MS of each sequence
        # MS = torch.zeros((DMS, 1), dtype=torch.float32)
        # seqC_abs = torch.abs(seqC.view(DMS, -1))
        # for i in range(DMS):
        #     seqC_i_abs = seqC_abs[i, :][~torch.isnan(seqC_abs[i, :])]
        #     MS[i] = torch.unique(seqC_i_abs[torch.nonzero(seqC_i_abs)])

        # compute MS of each sequence - faster implementation
        MS = torch.zeros((DMS, 1), dtype=torch.float32)
        seqC_abs = torch.abs(seqC)

        for i in range(15):
            # get the position of zero values of MS
            idx_zero = torch.nonzero(MS[:, 0] == 0)[:, 0]
            MS[idx_zero, :] = seqC_abs[idx_zero, i].unsqueeze(-1)

        return MS

    def _count_swtiches(self, seq):
        sign = torch.sign(seq)
        diff = torch.diff(sign)
        nSwt = torch.count_nonzero(diff)
        return nSwt

    def _compute_nSwitch(self, seqC, DMS):
        """
        compute number of switches NS
        [0, 1, 0, -1, 1, 0, 1, 0, 0, 1] -> 2

        Args:
            seqC: [D,M,S,15] or [DMS, 15]
        Returns:
        """
        # Prepare a tensor of zeros with the same shape as seqC
        # replace nan values with 0
        NS = torch.zeros((DMS, 1), dtype=torch.int)

        for i in range(DMS):
            seq = seqC[i, :]
            seq = seq[torch.nonzero(seq).squeeze(-1)]
            if len(seq) > 1:
                # Compute number of switches using torch.diff and torch.count_nonzero
                num_swtiches = self._count_swtiches(seq)
                NS[i, 0] = num_swtiches

        return NS

    def plot_kernels(self, save_fig=False, kernel=0, no_dist=False):
        D, M, S = self.D, self.M, self.S
        stats_MD, dist_MD, stats_MD2, dist_MD2, stats_NS, dist_NS, stats_psy = (
            self.stats_MD,
            self.dist_MD,
            self.stats_MD2,
            self.dist_MD2,
            self.stats_NS,
            self.dist_NS,
            self.stats_psy,
        )
        MD_list, Dur_list, MS_list = self.MD_list, self.Dur_list, self.MS_list

        # plot stats_MD  - feature 1&2
        if kernel == 0 or kernel == 1 or kernel == 2:
            fig, axs = plt.subplots(1, 3, figsize=(15, 7))
            axs = axs.flatten()
            for i in range(3):
                ax = axs[i]
                im = ax.imshow(
                    stats_MD[:, i, :].T.numpy(),
                    cmap="Blues",
                    interpolation="nearest",
                    vmin=stats_MD[:, 0, :].min(),
                    vmax=stats_MD[:, 0, :].max(),
                )
                ax.set_yticks(torch.arange(len(MD_list)))
                ax.set_yticklabels(MD_list.numpy())
                ax.set_xticks(torch.arange(D))
                ax.set_xticklabels(Dur_list.numpy())
                # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                ax.tick_params(axis="both", which="both", length=0)
                ax.set_title(f"\nMS={MS_list[i]:.2f}")
                ax.set_ylabel("MD")
                ax.set_xlabel("Dur")
                fig.tight_layout()
            cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5)
            cbar.ax.set_ylabel("probability", rotation=-90, va="bottom")
            if save_fig:
                fig.savefig(f"./src/dataset/figures/stats_MD.png", dpi=300)

            if not no_dist:
                # plot distribution MD  - feature 1&2
                fig, axs = plt.subplots(1, 3, figsize=(15, 7))
                axs = axs.flatten()
                for i in range(3):
                    ax = axs[i]
                    im = ax.imshow(
                        dist_MD[:, i, :].T.numpy(),
                        cmap="Blues",
                        interpolation="nearest",
                        vmin=dist_MD[:, 0, :].min(),
                        vmax=dist_MD[:, 0, :].max(),
                    )
                    ax.set_yticks(torch.arange(len(MD_list)))
                    ax.set_yticklabels(MD_list.numpy())
                    ax.set_xticks(torch.arange(D))
                    ax.set_xticklabels(Dur_list.numpy())
                    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                    ax.tick_params(axis="both", which="both", length=0)
                    ax.set_title(f"MD distribution\nMS={MS_list[i]:.2f}")
                    ax.set_ylabel("MD")
                    ax.set_xlabel("Dur")
                    fig.tight_layout()
                cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5)
                cbar.ax.set_ylabel("percentage", rotation=-90, va="bottom")
                if save_fig:
                    fig.savefig(f"./src/dataset/figures/dist_MD.png", dpi=300)

        if kernel == 0 or kernel == 3:
            # plot feature 3 - Right choice percentage (according to MS)
            fig = plt.figure(figsize=(15, 7))
            im = plt.imshow(stats_MD2.T.numpy(), cmap="Blues", interpolation="nearest")
            plt.yticks(torch.arange(len(MD_list)), MD_list.numpy())
            plt.xticks(torch.arange(M), MS_list.numpy())
            plt.ylabel("MD")
            plt.xlabel("MS")
            plt.title("Right choice percentage")
            cbar = fig.colorbar(im, shrink=0.5)
            cbar.ax.set_ylabel("probability", rotation=-90, va="bottom")
            if save_fig:
                fig.savefig(f"./src/dataset/figures/stats_MD2.png", dpi=300)

            if not no_dist:
                # plot feature 3 - distribution (according to MS)
                fig = plt.figure(figsize=(15, 7))
                im = plt.imshow(dist_MD2.T.numpy(), cmap="Blues", interpolation="nearest")
                plt.xticks(torch.arange(M), MS_list.numpy())
                plt.yticks(torch.arange(len(MD_list)), MD_list.numpy())
                plt.xlabel("MS")
                plt.ylabel("MD")
                plt.title("Distribution")
                cbar = fig.colorbar(im, shrink=0.5)
                cbar.ax.set_ylabel("percentage", rotation=-90, va="bottom")
                if save_fig:
                    fig.savefig(f"./src/dataset/figures/dist_MD2.png", dpi=300)

        if kernel == 0 or kernel == 4:
            # plot Right choice percentage (No switch condition) - feature 4
            fig, axs = plt.subplots(1, 3, figsize=(15, 7))
            axs = axs.flatten()
            for i in range(3):
                ax = axs[i]
                im = ax.imshow(
                    stats_NS[:, i, :].T.numpy(),
                    cmap="Blues",
                    interpolation="nearest",
                    vmin=stats_NS[:, 0, :].min(),
                    vmax=stats_NS[:, 0, :].max(),
                )
                ax.set_yticks(torch.arange(len(MD_list)))
                ax.set_xticks(torch.arange(D))
                ax.set_yticklabels(MD_list.numpy())
                ax.set_xticklabels(Dur_list.numpy())
                # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                ax.tick_params(axis="both", which="both", length=0)
                ax.set_title(f"Right choice percentage (No switch)\nMS={MS_list[i]:.2f}")
                ax.set_ylabel("MD")
                ax.set_xlabel("Dur")
                fig.tight_layout()
            cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5)
            cbar.ax.set_ylabel("probability", rotation=-90, va="bottom")
            if save_fig:
                fig.savefig(f"./src/dataset/figures/stats_NS.png", dpi=300)

            if not no_dist:
                # plot distribution NS - feature 4
                fig, axs = plt.subplots(1, 3, figsize=(15, 7))
                axs = axs.flatten()
                for i in range(3):
                    ax = axs[i]
                    im = ax.imshow(
                        dist_NS[:, i, :].T.numpy(),
                        cmap="Blues",
                        interpolation="nearest",
                        vmin=dist_NS[:, 0, :].min(),
                        vmax=dist_NS[:, 0, :].max(),
                    )
                    ax.set_yticks(torch.arange(len(MD_list)))
                    ax.set_xticks(torch.arange(D))
                    ax.set_yticklabels(MD_list.numpy())
                    ax.set_xticklabels(Dur_list.numpy())
                    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                    ax.tick_params(axis="both", which="both", length=0)
                    ax.set_title(f"NS distribution\nMS={MS_list[i]:.2f}")
                    ax.set_ylabel("MD")
                    ax.set_xlabel("Dur")
                    fig.tight_layout()
                cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5)
                cbar.ax.set_ylabel("percentage", rotation=-90, va="bottom")
                if save_fig:
                    fig.savefig(f"./src/dataset/figures/dist_NS.png", dpi=300)

        # plot feature 5 psy kernel
        if kernel == 0 or kernel == 5:
            fig, axs = plt.subplots(1, 3, figsize=(15, 7))
            axs = axs.flatten()
            for i in range(3):
                ax = axs[i]
                im = ax.imshow(
                    stats_psy[:, i, :].numpy(),
                    cmap="Blues",
                    interpolation="nearest",
                    vmin=stats_psy[:, 0, :].min(),
                    vmax=stats_psy[:, 0, :].max(),
                )
                ax.set_xticks(torch.arange(15 - 1))
                ax.set_yticks(torch.arange(len(Dur_list)))
                ax.set_yticklabels(Dur_list.numpy())
                # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                ax.tick_params(axis="both", which="both", length=0)
                ax.set_title(f"Right choice percentage (No switch)\nMS={MS_list[i]:.2f}")
                ax.set_ylabel("Dur")
                ax.set_xlabel("pulse position")
                fig.tight_layout()
            cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.3)
            cbar.ax.set_ylabel("probability", rotation=-90, va="bottom")
            if save_fig:
                fig.savefig(f"./src/dataset/figures/stats_psy.png", dpi=300)


def main():
    DATA_PATH = adapt_path("~tmp/NSC/data/dataset/dataset-L0-Eset0-100sets-T500.h5")
    idx_set = 0
    idx_theta = 10

    # =================================================== load h5 dataset file
    f = h5py.File(DATA_PATH, "r")
    """
    f has keys ['set_0', 'set_1', 'set_10', 'set_11']
    in one set, there are 3 keys: ['seqC', 'theta', 'probR']
    seqC:  [D, M, S, 15]            - [7, 3, 700, 15]
    theta: [T, 4]                   - [5000, 4]
    probR: [D, M, S, T, 1]          - [7, 3, 700, 5000, 1]
    """
    seqC = torch.from_numpy(f[f"set_{idx_set}"]["seqC"][:]).type(torch.float32)
    theta = torch.from_numpy(f[f"set_{idx_set}"]["theta"][idx_theta, :]).type(torch.float32)
    probR = torch.from_numpy(f[f"set_{idx_set}"]["probR"][:, :, :, idx_theta, :]).type(torch.float32)
    f.close()

    D, M, S = seqC.shape[0], seqC.shape[1], seqC.shape[2]
    DMS = D * M * S

    # sampling from probR
    probR = probR.reshape(DMS, -1)  # [D*M*S, 1]
    chR = torch.bernoulli(probR)  # [D*M*S, 1]
    # =================================================== load h5 dataset file

    # =================================================== load trials.mat
    sID = 2
    trials = sio.loadmat(adapt_path("~/tmp/NSC/data/trials.mat"))
    trials_data = trials["data"]
    trials_info = trials["info"]
    subjectID = torch.from_numpy(trials_data[0, -1])
    idx_subj = (subjectID == sID).squeeze(-1)
    chR = torch.from_numpy(trials_data[0, 42][idx_subj]).type(torch.float32)
    seqC = torch.from_numpy(trials_data[0, 0][idx_subj]).type(torch.float32)
    # =================================================== load trials.mat

    # generate kernels
    FG = Feature_Generator()
    FG.compute_kernels(seqC, chR, D, M, S)
    FG.plot_kernels(D, M, S)
    feature_1s, feature_2s, feature_3s, feature_4s, feature_5s = FG.get_features()
    feature = FG.compute_kernels(seqC, chR, D, M, S).get_provided_feature()
    FG.plot_kernels(D, M, S)

    figure = plt.figure(figsize=(22, 7))
    plt.plot(feature, ".-")
    plt.grid(alpha=0.2)
    plt.show()

    return feature_1s, feature_2s, feature_3s, feature_4s, feature_5s


def feature_gen_for_whole_dataset(data_path, feature_path):
    """generate feature for the whole dataset"""
    # data_path = "/home/wehe/tmp/NSC/data/dataset/dataset_L0_exp_0_set100_T500.h5"
    # feature_path = (
    #     "/home/wehe/tmp/NSC/data/dataset/feature_L0_exp_0_set100_T500_C100.h5"
    # )

    # remove feature file if exists
    if os.path.exists(feature_path):
        os.remove(feature_path)

    f = h5py.File(data_path, "r")
    f_feature = h5py.File(feature_path, "a")

    # with h5py.File(data_path, 'r') as f:
    sets = list(f.keys())

    C = 100
    for group_name, group in f.items():
        print(
            f"processing group: {group_name} [{sets.index(group_name)}/{len(sets)-1}]",
            end=" ",
        )

        seqC = torch.from_numpy(group["seqC"][:]).type(torch.float32)
        probR = torch.from_numpy(group["probR"][:]).type(torch.float32).to("cuda")
        theta = torch.from_numpy(group["theta"][:]).type(torch.float32)

        # sample probR to get chR, with 100 samples
        print(f"chR generated ", end="")
        start_time = time.time()
        chR = torch.repeat_interleave(probR, C, dim=-1)
        chR = torch.bernoulli(chR).to("cpu")  # [D,M,S,T,C]
        torch.cuda.empty_cache()
        D, M, S, T, C = chR.shape
        print(f"in {(time.time() - start_time)/60:.2f} min")

        feature_1s = torch.empty((T, C, M, 69))
        feature_2s = torch.empty((T, C, M, 69))
        feature_3s = torch.empty((T, C, M, 15))
        feature_4s = torch.empty((T, C, M, 12))
        feature_5s = torch.empty((T, C, M, 56))

        f_feature.create_group(group_name)
        f_feature[group_name].create_dataset("theta", data=theta)

        # generate kernels
        for idx_T in range(T):  # 500
            for idx_C in tqdm(range(C)):  # 100
                chR_ = chR[:, :, :, idx_T, idx_C].view(D, M, S, 1)  # [D,M,S,1]

                FG = Feature_Generator()
                (
                    feature_1,
                    feature_2,
                    feature_3,
                    feature_4,
                    feature_5,
                ) = FG.compute_kernels(seqC, chR_, D, M, S).get_features()

                feature_1s[idx_T, idx_C, :, :] = feature_1
                feature_2s[idx_T, idx_C, :, :] = feature_2
                feature_3s[idx_T, idx_C, :, :] = feature_3
                feature_4s[idx_T, idx_C, :, :] = feature_4
                feature_5s[idx_T, idx_C, :, :] = feature_5
            # break
        f_feature[group_name].create_dataset("feature_1", data=feature_1s)
        f_feature[group_name].create_dataset("feature_2", data=feature_2s)
        f_feature[group_name].create_dataset("feature_3", data=feature_3s)
        f_feature[group_name].create_dataset("feature_4", data=feature_4s)
        f_feature[group_name].create_dataset("feature_5", data=feature_5s)
        # break
    f.close()
    f_feature.close()


def compute_features(args):
    idx_T, idx_C, chR, seqC, D, M, S = args
    chR_ = chR[:, :, :, idx_T, idx_C].view(D, M, S, 1)  # [D,M,S,1]

    # Initialize the Feature_Generator
    FG = Feature_Generator()
    # Compute features
    features = FG.compute_kernels(seqC, chR_, D, M, S).get_features()

    # Return the index and features
    return idx_T, idx_C, features


def feature_gen_for_whole_dataset_parallel_for_one_set(data_path, feature_path, set_idx=0, debug=False):
    """generate feature for the whole dataset"""
    # data_path = "/home/wehe/tmp/NSC/data/dataset/dataset_L0_exp_0_set100_T500.h5"
    # feature_path = (
    #     "/home/wehe/tmp/NSC/data/dataset/feature_L0_exp_0_set100_T500_C100.h5"
    # )
    if debug:
        C = 2
    else:
        C = 100
    torch.set_num_threads(1)

    # remove feature file if exists
    if os.path.exists(feature_path):
        os.remove(feature_path)

    f = h5py.File(data_path, "r")
    f_feature = h5py.File(feature_path, "a")
    sets = list(f.keys())

    group_name = sets[set_idx]
    group = f[group_name]

    print(f"loading data from h5 file set [{group_name}]", end=" ")
    start_time = time.time()
    seqC = torch.from_numpy(group["seqC"][:]).type(torch.float32)
    probR = torch.from_numpy(group["probR"][:]).type(torch.float32)
    theta = torch.from_numpy(group["theta"][:]).type(torch.float32)
    print(f"in {(time.time() - start_time)/60:.2f} min")

    # sample probR to get chR, with 100 samples
    print(f"chR generating ... ", end="")
    start_time = time.time()
    chR = torch.repeat_interleave(probR, C, dim=-1)
    chR = torch.bernoulli(chR)  # [D,M,S,T,C]
    # torch.cuda.empty_cache()
    D, M, S, T, C = chR.shape
    print(f"in {(time.time() - start_time)/60:.2f} min")

    feature_1s = torch.empty((T, C, M, 69))
    feature_2s = torch.empty((T, C, M, 69))
    feature_3s = torch.empty((T, C, M, 15))
    feature_4s = torch.empty((T, C, M, 12))
    feature_5s = torch.empty((T, C, M, 56))

    f_feature.create_group(group_name)
    f_feature[group_name].create_dataset("theta", data=theta)

    print(f"{T*C} feature extraction tasks to be done, update time interval 20s")
    # Create a multiprocessing pool
    num_workers = min(32, os.cpu_count())
    print(f"{num_workers=}")
    pool = multiprocessing.Pool(processes=num_workers)

    # Submit all the tasks for execution
    args_list = [(idx_T, idx_C, chR, seqC, D, M, S) for idx_T in range(T) for idx_C in range(C)]

    for idx_T, idx_C, features in tqdm(
        pool.imap_unordered(compute_features, args_list),
        total=len(args_list),
        mininterval=20,
    ):
        feature_1s[idx_T, idx_C, :, :] = features[0]
        feature_2s[idx_T, idx_C, :, :] = features[1]
        feature_3s[idx_T, idx_C, :, :] = features[2]
        feature_4s[idx_T, idx_C, :, :] = features[3]
        feature_5s[idx_T, idx_C, :, :] = features[4]

    # Close the pool and wait for all the tasks to complete
    pool.close()
    pool.join()

    print("finished computing features, saving to h5 file")
    f_feature[group_name].create_dataset("feature_1", data=feature_1s)
    f_feature[group_name].create_dataset("feature_2", data=feature_2s)
    f_feature[group_name].create_dataset("feature_3", data=feature_3s)
    f_feature[group_name].create_dataset("feature_4", data=feature_4s)
    f_feature[group_name].create_dataset("feature_5", data=feature_5s)
    # break

    print("finished saving features to h5 file")
    f.close()
    f_feature.close()


if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()

    # main()
    data_path = "/home/ubuntu/tmp/NSC/data/dataset/dataset-L0-Eset0-100sets-T500v2.h5"
    feat_path = "/home/ubuntu/tmp/NSC/data/dataset/feature-L0-Eset0-100sets-T500v2-C100.h5"
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_idx", "-s", type=int, default=0)
    parser.add_argument("--data_path", "-data", type=str, default=data_path)
    parser.add_argument("--feat_path", "-feat", type=str, default=feat_path)
    args = parser.parse_args()

    set_idx = args.set_idx
    data_path = args.data_path
    feat_path = args.feat_path
    # file_name = f"feature-L0-Eset0-100sets-T500-C100-set{set_idx}.h5"
    # file_name = data_path.split("/")[-1].split(".")[0] + f"-C100-set{set_idx}.h5"

    # set_idx = 0
    # data_dir = "/".join(data_path.split("/")[:-1])
    # feat_path = os.path.join(data_dir, file_name)

    # feature_gen_for_whole_dataset(data_path, feature_path)
    feature_gen_for_whole_dataset_parallel_for_one_set(data_path, feat_path, set_idx=set_idx, debug=False)

    # feat_path = (
    #     "/home/wehe/tmp/NSC/data/dataset/feature_L0_exp_0_set100_T500_C100_set0.h5"
    # )
    # f = h5py.File(feat_path, "r")
    # f["set_0"].keys()
    # f["set_0"]["feature_1"].shape

    # profiler.disable()
    # profiler.print_stats(sort="time")
