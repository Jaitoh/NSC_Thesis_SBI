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

DATA_PATH = "/mnt/data/dataset/dataset_L0_exp_set_0.h5"
idx_set   = 0
idx_theta = 10

# load h5 dataset file
f = h5py.File(DATA_PATH, 'r')
"""
f has keys ['set_0', 'set_1', 'set_10', 'set_11']
in one set, there are 3 keys: ['seqC', 'theta', 'probR']
seqC:  [D, M, S, 15]            - [7, 3, 700, 15]
theta: [T, 4]                   - [5000, 4]
probR: [D, M, S, T, 1]          - [7, 3, 700, 5000, 1]
"""
seqC  = torch.from_numpy(f[f'set_{idx_set}']['seqC'][:]).type(torch.float32)
theta = torch.from_numpy(f[f'set_{idx_set}']['theta'][idx_theta, :]).type(torch.float32)
probR = torch.from_numpy(f[f'set_{idx_set}']['probR'][:, :, :, idx_theta, :]).type(torch.float32)
f.close()

D, M, S = seqC.shape[0], seqC.shape[1], seqC.shape[2]
DMS = D*M*S

# sampling from probR
probR = probR.reshape(DMS, -1) # [D*M*S, 1]
chR = torch.bernoulli(probR) # [D*M*S, 1]

# # ==================================================================== load trials.mat
sID = 2
trials = sio.loadmat('/home/ubuntu/tmp/NSC/data/trials.mat')
trials_data = trials['data']
trials_info = trials['info']
subjectID = torch.from_numpy(trials_data[0, -1])
idx_subj = (subjectID == sID).squeeze(-1)
chR = torch.from_numpy(trials_data[0, 42][idx_subj]).type(torch.float32)
seqC = torch.from_numpy(trials_data[0, 0][idx_subj]).type(torch.float32)
# # ==================================================================== load trials.mat

# compute Dur of each sequence
Dur = torch.sum(~torch.isnan(seqC), dim=-1, dtype=torch.int32).view(D*M*S, 1) # [DMS, 1]

def get_MS(seqC, DMS):
    
    # # compute MS of each sequence
    # MS = torch.zeros((DMS, 1), dtype=torch.float32)
    # seqC_abs = torch.abs(seqC.view(DMS, -1))
    
    # for i in range(DMS):
    #     seqC_i_abs = seqC_abs[i, :][~torch.isnan(seqC_abs[i, :])]
    #     MS[i] = torch.unique(seqC_i_abs[torch.nonzero(seqC_i_abs)])
    
    # compute MS of each sequence - faster implementation
    MS = torch.zeros((DMS, 1), dtype=torch.float32)
    seqC_abs = torch.abs(seqC.view(DMS, -1)).nan_to_num()

    for i in range(15):
        # get the position of zero values of MS
        idx_zero = torch.nonzero(MS[:, 0]==0)[:,0]
        MS[idx_zero, :] = seqC_abs[idx_zero, i].unsqueeze(-1)
        
    return MS

MS = get_MS(seqC, DMS) # [DMS, 1]

# compute MD
MD = torch.sum(torch.sign(seqC), axis=-1, dtype=torch.float32).view(DMS, -1) # [D, M, S, 1]

# compute net MD
seqC_2D = seqC.view(DMS, -1) # [D*M*S, 15]

def count_swtiches(seq):
    sign = torch.sign(seq)
    diff = torch.diff(sign)
    nSwt = torch.count_nonzero(diff)
    return nSwt
    
def compute_ns(seqC, DMS):
    """    
    compute number of switches NS
    seqC: D,M,S,15
    when seqC[0,0,0,:] all >=0 or seqC[0,0,0,:] all <=0
    NS[0,0,0, 0] = 1, else 0
        
    Args:
        seqC: D,M,S,15
        when seqC[0,0,0,:] all >=0 or seqC[0,0,0,:] all <=0
        NS[0,0,0, 0] = 1, else 0
    Returns:
    """
    # Prepare a tensor of zeros with the same shape as seqC
    # replace nan values with 0
    seqC[torch.isnan(seqC)] = 0
    # D, M, S = seqC.shape[0], seqC.shape[1], seqC.shape[2]
    seqC = seqC.view(DMS, -1)
    NS = torch.zeros((DMS, 1), dtype=torch.int)
    
    for i in range(DMS):
        seq = seqC[i, :]
        seq = seq[torch.nonzero(seq).squeeze(-1)]
        if len(seq) >1:
            # Compute number of switches using torch.diff and torch.count_nonzero
            num_swtiches = count_swtiches(seq)
            NS[i, 0] = num_swtiches
                    
    return NS

# compute NS 
nSwitch = compute_ns(seqC, DMS).view(DMS, -1) # [DMS, 1]

# dataframe for visualization
data = pd.DataFrame()
data['probR']   = probR[:, 0]
data['chR']     = chR[:, 0]
data['Dur']     = Dur[:, 0]
data['MS']      = MS[:, 0]
data['MD']      = MD[:, 0]
data['NSwitch'] = nSwitch[:, 0]


Dur_list = torch.unique(Dur, sorted=True)
MS_list = torch.unique(MS, sorted=True)
# MD_list = torch.unique(MD, sorted=True)
MD_list = torch.range(-10, 10)


dist_MD     = torch.zeros((D, M, len(MD_list)), dtype=torch.float32) # for feature 1&2
dist_MD2    = torch.zeros((M, len(MD_list)), dtype=torch.float32)    # for feature 3
dist_NS     = torch.zeros((D, M, len(MD_list)), dtype=torch.float32) # for feature 4

stats_MD    = torch.zeros((D, M, len(MD_list)), dtype=torch.float32)
stats_MD2   = torch.zeros((M, len(MD_list)), dtype=torch.float32)
stats_NS    = torch.zeros((D, M, len(MD_list)), dtype=torch.float32)

# compute feature 1&2, 4
for idx_D in range(D):
    idx_current_D  = (Dur == Dur_list[idx_D]).squeeze()
    
    for idx_M in range(M):
        idx_current_M  = (MS  == MS_list[idx_M]).squeeze()
        
        for idx_MD in range(len(MD_list)):
            
            # idx_D, idx_M, idx_MD = 0, 0, 9
            idx_current_MD = (MD  == MD_list[idx_MD]).squeeze()
            idx_current_NS = (nSwitch == 0).squeeze()
            
            # feature 1&2
            idx_f12 = idx_current_D & idx_current_M & idx_current_MD
            if torch.sum(idx_f12) != 0:
                # compute the stats for the current MD
                chR_chosen = chR[idx_f12, :]
                stats_MD[idx_D, idx_M, idx_MD]  = torch.mean(chR_chosen)
                dist_MD[idx_D, idx_M, idx_MD]   = torch.sum(idx_f12) / S
            
            # feature 4
            idx_f4 = idx_current_D & idx_current_M & idx_current_MD & idx_current_NS
            if torch.sum(idx_f4) != 0:
                chR_chosen = chR[idx_f4, :]
                stats_NS[idx_D, idx_M, idx_MD]  = torch.mean(chR_chosen)
                dist_NS[idx_D, idx_M, idx_MD]   = torch.sum(chR_chosen) / S



# compute feature 3
for idx_M in range(M):
    idx_current_M  = (MS  == MS_list[idx_M]).squeeze()
    
    for idx_MD in range(len(MD_list)):
        idx_current_MD = (MD  == MD_list[idx_MD]).squeeze()
        
        # feature 3
        idx_f3 = idx_current_M & idx_current_MD
        if torch.sum(idx_f3) != 0:
            chR_chosen = chR[idx_f3, :]
            stats_MD2[idx_M, idx_MD]  = torch.mean(chR_chosen)
            dist_MD2[idx_M, idx_MD]   = torch.sum(chR_chosen) / (D*S)

# compute feature 5
stats_psy_p = torch.zeros((D, M, 15-1), dtype=torch.float32)
stats_psy_n = torch.zeros((D, M, 15-1), dtype=torch.float32)

for idx_D in range(D):
    idx_current_D = (Dur == Dur_list[idx_D]).squeeze()
    
    for idx_M in range(M):
        idx_current_M  = (MS  == MS_list[idx_M]).squeeze()
    
        for idx_P in range(1, 15): # pulse position
            idx_current_pP = seqC_2D[:, idx_P]>0 # positive pulse position
            idx_current_nP = seqC_2D[:, idx_P]<0 # negative pulse position
            
            idx_f5_p = idx_current_D & idx_current_M & idx_current_pP
            chR_chosen = chR[idx_f5_p, :]
            stats_psy_p[idx_D, idx_M, idx_P-1]  = torch.mean(chR_chosen)
            
            idx_f5_n = idx_current_D & idx_current_M & idx_current_nP
            chR_chosen = chR[idx_f5_n, :]
            stats_psy_n[idx_D, idx_M, idx_P-1]  = torch.mean(chR_chosen)

stats_psy = stats_psy_p - stats_psy_n
stats_psy.nan_to_num_(0)


# ============================================================================= plots
# TODO: remove .T
# plot stats_MD  - feature 1&2
fig, axs = plt.subplots(1, 3, figsize=(25, 7))
axs = axs.flatten()
for i in range(3):
    ax = axs[i]
    im = ax.imshow(stats_MD[:, i, :].numpy(), cmap='Blues', interpolation='nearest', vmin=stats_MD[:, 0, :].min(), vmax=stats_MD[:, 0, :].max())
    ax.set_xticks(torch.arange(len(MD_list)))
    ax.set_xticklabels(MD_list.numpy())
    ax.set_yticks(torch.arange(D))
    ax.set_yticklabels(Dur_list.numpy())
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis=u'both', which=u'both', length=0)
    ax.set_title(f"Right choice percentage\nMS={MS_list[i]:.2f}")
    ax.set_ylabel("MD")
    ax.set_xlabel("Dur")
    fig.tight_layout()
cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5)
cbar.ax.set_ylabel("probability", rotation=-90, va="bottom")
plt.show()

# plot distribution MD  - feature 1&2
fig, axs = plt.subplots(1, 3, figsize=(15, 7))
axs = axs.flatten()
for i in range(3):
    ax = axs[i]
    im = ax.imshow(dist_MD[:, i, :].T.numpy(), cmap='Blues', interpolation='nearest', vmin=dist_MD[:, 0, :].min(), vmax=dist_MD[:, 0, :].max())
    ax.set_yticks(torch.arange(len(MD_list)))
    ax.set_xticks(torch.arange(D))
    ax.set_yticklabels(MD_list.numpy())
    ax.set_xticklabels(Dur_list.numpy())
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis=u'both', which=u'both', length=0)
    ax.set_title(f"MD distribution\nMS={MS_list[i]:.2f}")
    ax.set_ylabel("MD")
    ax.set_xlabel("Dur")
    fig.tight_layout()
cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5)
cbar.ax.set_ylabel("percentage", rotation=-90, va="bottom")
plt.show()

# plot feature 3 - Right choice percentage (according to MS)
fig = plt.figure(figsize=(15, 7))
im = plt.imshow(stats_MD2.T.numpy(), cmap='Blues', interpolation='nearest')
plt.xticks(torch.arange(M), MS_list.numpy())
plt.yticks(torch.arange(len(MD_list)), MD_list.numpy())
plt.xlabel("MS")
plt.ylabel("MD")
plt.title("Right choice percentage")
cbar = fig.colorbar(im, shrink=0.5)
cbar.ax.set_ylabel("probability", rotation=-90, va="bottom")
plt.show()

# plot feature 3 - distribution (according to MS)
fig = plt.figure(figsize=(15, 7))
im = plt.imshow(dist_MD2.T.numpy(), cmap='Blues', interpolation='nearest')
plt.xticks(torch.arange(M), MS_list.numpy())
plt.yticks(torch.arange(len(MD_list)), MD_list.numpy())
plt.xlabel("MS")
plt.ylabel("MD")
plt.title("Distribution")
cbar = fig.colorbar(im, shrink=0.5)
cbar.ax.set_ylabel("percentage", rotation=-90, va="bottom")
plt.show()


# plot Right choice percentage (No switch condition) - feature 4
fig, axs = plt.subplots(1, 3, figsize=(15, 7))
axs = axs.flatten()
for i in range(3):
    ax = axs[i]
    im = ax.imshow(stats_NS[:, i, :].T.numpy(), cmap='Blues', interpolation='nearest', vmin=stats_NS[:, 0, :].min(), vmax=stats_NS[:, 0, :].max())
    ax.set_yticks(torch.arange(len(MD_list)))
    ax.set_xticks(torch.arange(D))
    ax.set_yticklabels(MD_list.numpy())
    ax.set_xticklabels(Dur_list.numpy())
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis=u'both', which=u'both', length=0)
    ax.set_title(f"Right choice percentage (No switch)\nMS={MS_list[i]:.2f}")
    ax.set_ylabel("MD")
    ax.set_xlabel("Dur")
    fig.tight_layout()
cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5)
cbar.ax.set_ylabel("probability", rotation=-90, va="bottom")
plt.show()

# plot distribution NS - feature 4
fig, axs = plt.subplots(1, 3, figsize=(15, 7))
axs = axs.flatten()
for i in range(3):
    ax = axs[i]
    im = ax.imshow(dist_NS[:, i, :].T.numpy(), cmap='Blues', interpolation='nearest', vmin=dist_NS[:, 0, :].min(), vmax=dist_NS[:, 0, :].max())
    ax.set_yticks(torch.arange(len(MD_list)))
    ax.set_xticks(torch.arange(D))
    ax.set_yticklabels(MD_list.numpy())
    ax.set_xticklabels(Dur_list.numpy())
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis=u'both', which=u'both', length=0)
    ax.set_title(f"NS distribution\nMS={MS_list[i]:.2f}")
    ax.set_ylabel("MD")
    ax.set_xlabel("Dur")
    fig.tight_layout()
cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5)
cbar.ax.set_ylabel("percentage", rotation=-90, va="bottom")
plt.show()

# plot feature 5 psy kernel
fig, axs = plt.subplots(1, 3, figsize=(15, 7))
axs = axs.flatten()
for i in range(3):
    ax = axs[i]
    im = ax.imshow(stats_psy[:, i, :].numpy(), cmap='Blues', interpolation='nearest', vmin=stats_psy[:, 0, :].min(), vmax=stats_psy[:, 0, :].max())
    ax.set_xticks(torch.arange(15-1))
    # ax.set_xticklabels(Dur_list.numpy())
    ax.set_yticks(torch.arange(len(Dur_list)))
    ax.set_yticklabels(Dur_list.numpy())
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis=u'both', which=u'both', length=0)
    ax.set_title(f"Right choice percentage (No switch)\nMS={MS_list[i]:.2f}")
    ax.set_ylabel("Dur")
    ax.set_xlabel("pulse position")
    fig.tight_layout()
cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5)
cbar.ax.set_ylabel("probability", rotation=-90, va="bottom")
plt.show()