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
inf_result_path = "~/tmp/NSC/codes/src/train/logs/train_L0_p5a/p5a-conv_net/inference/subj_thetas.pkl"
# load pkl
with open(adapt_path(inf_result_path), "rb") as f:
    inf_result = pickle.load(f)


subj_IDs = list(inf_result.keys())[3:]
exp = inf_result["exp"]
bias = [inf_result[subj_ID][0] for subj_ID in subj_IDs]
sigma2a = [inf_result[subj_ID][1] for subj_ID in subj_IDs]
sigma2s = [inf_result[subj_ID][2] for subj_ID in subj_IDs]
L0 = [inf_result[subj_ID][3] for subj_ID in subj_IDs]


# parameters infer
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
ax.plot(subj_IDs, bias, "ko-", label=exp_label)
ax.plot(subj_IDs, bias_fitted * 10, "ko--", label="fitted * 10", mfc="none")
ax.legend()
ax.set_xlabel("subj_ID")
ax.set_ylabel("bias")
ax.set_title("bias")
ax.grid(alpha=0.2)

ax = axes[0, 1]
axes[0, 1].plot(subj_IDs, L0, "ko-", label=exp_label)
axes[0, 1].plot(subj_IDs, L0_fitted * 1, "ko--", label="fitted", mfc="none")
ax.legend()
ax.set_xlabel("subj_ID")
ax.set_ylabel("L0")
ax.set_title("L0")
ax.grid(alpha=0.2)

ax = axes[1, 0]
axes[1, 0].plot(subj_IDs, sigma2a, "ko-", label=exp_label)
axes[1, 0].plot(subj_IDs, sigma2a_fitted * 1, "ko--", label="fitted", mfc="none")
ax.legend()
ax.set_xlabel("subj_ID")
ax.set_ylabel("sigma2a")
ax.set_title("sigma2a")
ax.grid(alpha=0.2)

ax = axes[1, 1]
axes[1, 1].plot(subj_IDs, sigma2s, "ko-", label=exp_label)
axes[1, 1].plot(subj_IDs, sigma2s_fitted * 1, "ko--", label="fitted", mfc="none")
ax.legend()
ax.set_xlabel("subj_ID")
ax.set_ylabel("sigma2s")
ax.set_title("sigma2s")
ax.grid(alpha=0.2)

# %%
