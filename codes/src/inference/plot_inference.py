# %%
import numpy as np
import pickle
from matplotlib import pyplot as plt
from pathlib import Path
import sys

NSC_DIR = Path(__file__).resolve().parent.parent.parent.parent.as_posix()  # NSC dir
sys.path.append(f"{NSC_DIR}/codes/src")

from utils.setup import adapt_path
from utils.subject import decode_mat_fitted_parameters, get_fitted_param_L0

import matplotlib as mpl

# remove top and right axis from plots
mpl.rcParams["axes.spines.right"] = True
mpl.rcParams["axes.spines.top"] = True
# remove all edges
mpl.rcParams["axes.edgecolor"] = "k"
mpl.rcParams["axes.linewidth"] = 4
font = {
    "weight": "bold",
    "size": 22,
}
mpl.rc("font", **font)
# title font
mpl.rcParams["axes.titlesize"] = font["size"]
mpl.rcParams["axes.titleweight"] = "bold"
# legend font
mpl.rcParams["legend.fontsize"] = 16
mpl.rcParams["legend.title_fontsize"] = 16
mpl.rcParams["legend.frameon"] = False
# set legend font to not bold


# %%
exps = ["train_L0_p5a/p5a-conv_net", "train_L0_p5a/p5a-conv_net-Tv2"]
bias_s, sigma2a_s, sigma2s_s, L0_s = [], [], [], []
for exp in exps:
    inf_result_path = f"~/tmp/NSC/codes/src/train/logs/{exp}/inference/subj_thetas.pkl"
    # load pkl
    with open(adapt_path(inf_result_path), "rb") as f:
        inf_result = pickle.load(f)

    subj_IDs = list(inf_result.keys())[3:]
    exp = inf_result["exp"]
    bias = [inf_result[subj_ID][0] for subj_ID in subj_IDs]
    sigma2a = [inf_result[subj_ID][1] for subj_ID in subj_IDs]
    sigma2s = [inf_result[subj_ID][2] for subj_ID in subj_IDs]
    L0 = [inf_result[subj_ID][3] for subj_ID in subj_IDs]

    bias_s.append(bias)
    sigma2a_s.append(sigma2a)
    sigma2s_s.append(sigma2s)
    L0_s.append(L0)


# %%
# get fitted parameters
param_path = "~/tmp/NSC/data/params/263 models fitPars/"
bias_fitted, sigma2a_fitted, sigma2s_fitted, L0_fitted = [], [], [], []
for subj_ID in subj_IDs:
    params_fitted = get_fitted_param_L0(param_path, subj_ID)
    bias_fitted.append(params_fitted[0])
    sigma2a_fitted.append(params_fitted[1])
    sigma2s_fitted.append(params_fitted[2])
    L0_fitted.append(params_fitted[3])
bias_fitted = np.array(bias_fitted)
sigma2a_fitted = np.array(sigma2a_fitted)
sigma2s_fitted = np.array(sigma2s_fitted)
L0_fitted = np.array(L0_fitted)

# %%
fig, axes = plt.subplots(2, 2, figsize=(22, 16))
# vertical space
fig.subplots_adjust(hspace=0.4, wspace=0.3)

exp_label = "/".join(exp)

ax = axes[0, 0]
for i in range(len(exps)):
    ax.plot(subj_IDs, bias_s[i], "o-", label=exps[i])
ax.plot(subj_IDs, bias_fitted * 10, "ko--", label="fitted * 10", mfc="none")
ax.legend()
ax.set_xlabel("subj_ID")
ax.set_ylabel("bias")
ax.set_title("bias")
ax.grid(alpha=0.2)

ax = axes[0, 1]
for i in range(len(exps)):
    ax.plot(subj_IDs, L0_s[i], "o-", label=exps[i])
ax.plot(subj_IDs, L0_fitted * 1, "ko--", label="fitted", mfc="none")
ax.legend()
ax.set_xlabel("subj_ID")
ax.set_ylabel("L0")
ax.set_title("L0")
ax.grid(alpha=0.2)

ax = axes[1, 0]
for i in range(len(exps)):
    ax.plot(subj_IDs, sigma2a_s[i], "o-", label=exps[i])
ax.plot(subj_IDs, sigma2a_fitted * 1, "ko--", label="fitted", mfc="none")
ax.legend()
ax.set_xlabel("subj_ID")
ax.set_ylabel("sigma2a")
ax.set_title("sigma2a")
ax.grid(alpha=0.2)

ax = axes[1, 1]
for i in range(len(exps)):
    ax.plot(subj_IDs, sigma2s_s[i], "o-", label=exps[i])
ax.plot(subj_IDs, sigma2s_fitted * 1, "ko--", label="fitted", mfc="none")
ax.legend()
ax.set_xlabel("subj_ID")
ax.set_ylabel("sigma2s")
ax.set_title("sigma2s")
ax.grid(alpha=0.2)

# %% t-SNE clustering for all parameters
# project fitted parameters to 2D using t-SNE
from sklearn.manifold import TSNE

# X = np.vstack([bias_fitted, sigma2a_fitted, sigma2s_fitted, L0_fitted]).T
X_embedded_s = []
for i in range(len(exps)):
    X = np.vstack([bias_s[i], sigma2a_s[i], sigma2s_s[i], L0_s[i]]).T
    X_embedded = TSNE(n_components=2, perplexity=5, random_state=0).fit_transform(X)
    X_embedded_s.append(X_embedded)

plt.figure(figsize=(8, 8))
for i in range(len(exps)):
    plt.scatter(X_embedded_s[i][:, 0], X_embedded_s[i][:, 1], s=50, label=exps[i])

    for j, subj_ID in enumerate(subj_IDs):
        plt.text(
            X_embedded_s[i][j, 0] + 0.5,
            X_embedded_s[i][j, 1] + 0,
            subj_ID,
            fontsize=10,
            color="k",
            alpha=0.5,
            verticalalignment="center",
            horizontalalignment="left",
        )

plt.legend()
plt.title(f"t-SNE clustering of subjects")
plt.grid(alpha=0.2)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()

# %% bias and L0 clustering
plt.figure(figsize=(8, 8))
for i in range(len(exps)):
    plt.scatter(bias_s[i], L0_s[i], s=50, label=exps[i])
    for j, subj_ID in enumerate(subj_IDs):
        plt.text(
            bias_s[i][j] + 0.01,
            L0_s[i][j] + 0.01,
            subj_ID,
            fontsize=10,
            color="k",
            alpha=0.5,
            verticalalignment="center",
            # horizontalalignment="center",
        )


plt.legend()

plt.title(f"L0&bias clustering of subjects")
plt.grid(alpha=0.2)
plt.xlabel("bias")
plt.ylabel("L0")
plt.show()

# %%
