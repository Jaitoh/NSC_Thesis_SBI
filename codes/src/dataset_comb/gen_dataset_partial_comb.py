import os
import pickle
import numpy as np
import argparse
from tqdm import tqdm
import sys
import h5py
from pathlib import Path
import time

NSC_DIR = Path(__file__).resolve().parent.parent.parent.parent.as_posix()  # NSC dir
sys.path.append(f"{NSC_DIR}/codes/src")

from simulator.model_sim_pR import (
    DM_sim_for_seqCs_parallel_with_smaller_output as sim_parallel,
)
from simulator.model_sim_pR import get_boxUni_prior
from simulator.seqC_generator import seqC_combinatorial_generator
from utils.set_seed import setup_seed
from utils.dataset.dataset import pad_seqC_with_nans_to_len15
from utils.setup import adapt_path

setup_seed(100)

# generate seqCs

debug = False

dur_list = [3, 5, 7, 9, 11, 13, 15]
num_chosen_seqC = [9 - 1, 81 - 1, 729 - 1, 6561 - 1, 6000, 6000, 6000]

if debug:
    dur_list = [3, 5, 7]
    num_chosen_seqC = [9 - 1, 81 - 1, 729 - 1]

do_generate_seqC_partial_combinatorial = True
if do_generate_seqC_partial_combinatorial:
    if os.path.exists(f"{NSC_DIR}/data/dataset-comb/seqC_partial_combinatorial.h5"):
        os.remove(f"{NSC_DIR}/data/dataset-comb/seqC_partial_combinatorial.h5")

    for i in tqdm(range(len(dur_list))):
        dur = dur_list[i]
        num_chosen = num_chosen_seqC[i]

        seqCs = seqC_combinatorial_generator(dur)

        # randomly choose the seqCs from shape (n, dur) into shape (num_chosen_seqC, dur)
        seqCs_chosen = seqCs[np.random.choice(len(seqCs), num_chosen, replace=False)]

        # save the output
        key = f"dur_{dur}"
        # if the file exists, delete it first

        with h5py.File(f"{NSC_DIR}/data/dataset-comb/seqC_partial_combinatorial.h5", "a") as f:
            f.create_dataset(key, data=seqCs_chosen)

    print(f"file saved to {NSC_DIR}/data/dataset-comb/seqC_partial_combinatorial.h5")

# load seqCs
f = h5py.File(f"{NSC_DIR}/data/dataset-comb/seqC_partial_combinatorial.h5", "r")
for key in f.keys():
    print(f"{key} number of stored combinations: {len(f[key][:]):7}")

# sample 500 samples from the prior range
num_prior_sample = 500
prior_min = [-2.5, 0, 0, -11]
prior_max = [2.5, 77, 18, 10]
prior = get_boxUni_prior(prior_min, prior_max)

ms_list = [0.2, 0.4, 0.8]

for dur in dur_list:
    # dur = 3
    time_start = time.time()
    seqCs_combs_dur = f[f"dur_{dur}"][:]
    print("".center(50, "-"))
    print(f"=== calculating probR for each seqC of dur_{dur} ===")
    print("".center(50, "-"))

    seqs = np.empty((1, len(ms_list), *seqCs_combs_dur.shape))  # shape (D, len(ms_list), num_seqC, dur)
    for i, ms in enumerate(ms_list):
        seqs[:, i, :] = f[f"dur_{dur}"][:] * ms_list[i]
    seqs = pad_seqC_with_nans_to_len15(seqs)

    num_seqC_dur = seqCs_combs_dur.shape[0]
    params_collect = np.zeros((1, len(ms_list), num_seqC_dur, num_prior_sample, 4))
    probR_collect = np.zeros((1, len(ms_list), num_seqC_dur, num_prior_sample, 1))

    for i in range(seqs.shape[1]):  # for each ms
        for j in tqdm(range(seqs.shape[2])):  # for each seqC
            # get one seqC and simulate with 500 samples
            seq = seqs[:, i, j, :][np.newaxis, np.newaxis, ...]
            # print(i, j, seq)
            params, probR = sim_parallel(
                seqCs=seq,
                prior=prior,
                num_prior_sample=num_prior_sample,
                model_name="B-G-L0S-O-N-",
                num_workers=16,
                privided_prior=False,
                verbose=0,
            )

            params_collect[0, i, j, :, :] = params
            probR_collect[:, i, j, :, :] = probR[:, 0, 0, :, :]
            if debug:
                break

    print(f"seqs.shape: {seqs.shape}")
    print(f"theta.shape: {params_collect.shape}")
    print(f"probR.shape: {probR_collect.shape}")
    print(f"--- time elapsed: {time.time() - time_start} seconds ---\n")

    output_dir = f"{NSC_DIR}/data/dataset-comb/dataset-partcomb-T500.h5"
    with h5py.File(output_dir, "a") as f_result:
        dur_group = f_result.create_group(f"dur_{dur}")
        dur_group.create_dataset("seqC", data=seqs)
        dur_group.create_dataset("theta", data=params_collect)
        dur_group.create_dataset("probR", data=probR_collect)

    print(f"dur_{dur} simulation data saved to {output_dir}\n")

f.close()

# check
# output_dir = f"{NSC_DIR}/data/dataset-comb/dataset-partcomb-T500.h5"
# f = h5py.File(output_dir, "r")
# f.keys()
# f["dur_5"].keys()
# f["dur_3"]["probR"][:].shape
# f["dur_3"]["probR"][0, 0, 2, :]
# f["dur_3"]["probR"][0, 0, 0, :]
# f["dur_5"]["theta"][:].shape
# f.close()
