"""
extracte features from / make summarization of 
seqC and cR

"""
import h5py
import os
import argparse

# data_dir = "/home/wehe/tmp/NSC/data/dataset/"
data_dir = "/home/wehe/scratch/data/feature/v2"
merged_data_path = "feature-L0-Eset0-98sets-T500v2-C100.h5"

parser = argparse.ArgumentParser(description="merge features")
parser.add_argument(
    "--data_dir",
    type=str,
    default=data_dir,
    help="simulated feature subsets store/load dir",
)
parser.add_argument(
    "--merged_data_path",
    type=str,
    default=merged_data_path,
)
args = parser.parse_args()

data_dir = args.data_dir
merged_data_path = args.merged_data_path

# list all files in the data_dir starting with "feature"
feature_files = [
    os.path.join(data_dir, f)
    for f in os.listdir(data_dir)
    if os.path.isfile(os.path.join(data_dir, f)) and f.startswith("feature")
]
feature_files.sort()
print(feature_files)

# dest_file = os.path.join(data_dir, merged_data_path)
dest_file = merged_data_path

f_dest = h5py.File(dest_file, "a")
for i in range(len(feature_files)):
    print(f"copying {feature_files[i]} [{i}/{len(feature_files)-1}] ... ", end="")
    f = h5py.File(feature_files[i], "r")
    group_name = list(f.keys())[0]
    f_dest.copy(f[group_name], group_name)
    f.close()
    print(f"done.")

f_dest.close()
