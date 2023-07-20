import pickle
import numpy as np
import argparse
from tqdm import tqdm
import sys
import h5py
from pathlib import Path

sys.path.append("./src")

from simulator.model_sim_pR import (
    DM_sim_for_seqCs_parallel_with_smaller_output as sim_parallel,
)
from simulator.model_sim_pR import get_boxUni_prior
from utils.set_seed import setup_seed
from simulator.seqC_generator import seqC_combinatorial_generator

NSC_DIR = Path(__file__).resolve().parent.parent.parent.parent.as_posix()  # NSC dir

setup_seed(100)

do_generate_seqC_combinatorial = False
if do_generate_seqC_combinatorial:
    output_seqs = {}

    dur_list = range(3, 15 + 1, 2)
    ms_list = [0.2, 0.4, 0.8]

    for dur in tqdm(dur_list):
        seqCs = seqC_combinatorial_generator(dur)
        key = f"dur_{dur}"

        # save the output
        with h5py.File(f"{NSC_DIR}/data/seqC_combinatorial.h5", "a") as f:
            f.create_dataset(key, data=seqCs)

    print(f"file saved to {NSC_DIR}/data/seqC_combinatorial.h5")

# get input arguments
args = argparse.ArgumentParser()
args.add_argument("--dur_list", type=str, default="[3,5,7,9,11,13,15]")
args.add_argument("--task_part", type=str, default="[0, 1]")
args = args.parse_args()

# dur_list = range(3, 15+1, 2)
dur_list = eval(args.dur_list)
dur_list = [3, 5, 7, 9, 11]
task_part = eval(args.task_part)
print(f"dur_list: {dur_list}")
print(f"task_part: {task_part}")
ms_list = [0.2, 0.4, 0.8]


with h5py.File(f"{NSC_DIR}/data/seqC_combinatorial.h5", "r") as f:
    for key in f.keys():
        print(f"{key} number of possible combinations: {len(f[key][:]):7}")

f = h5py.File(f"{NSC_DIR}/data/seqC_combinatorial.h5", "r")

# prior
num_prior_sample = 500
prior_min = [-2.5, 0, 0, -11]
prior_max = [2.5, 77, 18, 10]

prior = get_boxUni_prior(prior_min, prior_max)
# ! keep the same prior for all seqCs
params = prior.sample((num_prior_sample,)).cpu().numpy()

# reshape for parallel probR computation
for dur in dur_list:
    seqCs_combs_dur = f[f"dur_{dur}"][:]
    print(f"\n=== calculating probR for each seqC of dur_{dur}...")
    seqs = np.empty((1, 3, *seqCs_combs_dur.shape))
    for i, ms in enumerate(ms_list):
        seqs[:, i, :] = f[f"dur_{dur}"][:] * ms_list[i]

    # num_parts = 100 if dur > 8 else 1
    num_parts = 100 if dur > 11 else 1
    # subdivide seqs into 100 chunks along the 3rd axis
    seqs_divided = np.array_split(seqs, num_parts, axis=2)
    info = f"reshaped seqs of size: {seqs.shape}\nsubdivide into {len(seqs_divided)} chunks along the 3rd axis\none divided seqs has size: {seqs_divided[0].shape}"
    print(info)

    task_nums = np.arange(task_part[0], task_part[1])
    for i, seq in enumerate(seqs_divided[task_part[0] : task_part[1]]):
        print(f"\nchunk {task_nums[i]} of {task_part[0]} to {task_part[1]}")

        params, probR = sim_parallel(
            seqCs=seq,
            prior=params,
            num_prior_sample=num_prior_sample,
            model_name="B-G-L0S-O-N-",
            num_workers=16,
            privided_prior=True,
        )

        output_dir = (
            f"{NSC_DIR}/data/dataset_combinatorial_dur_{dur}_part{task_nums[i]}.h5"
        )
        with h5py.File(output_dir, "w") as f_result:
            f_result.create_dataset(f"seqC", data=seq)
            f_result.create_dataset(f"theta", data=params)
            f_result.create_dataset(f"probR", data=probR)

        print(f"file saved to {output_dir}\n")


f.close()
