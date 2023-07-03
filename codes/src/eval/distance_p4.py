from pathlib import Path
from omegaconf import OmegaConf
import sys
import torch

sys.path.append("./src")
sys.path.append("../../src")

from train.train_L0_p4 import Solver

# compute the distance between the predicted and the ground truth
EXP_ID = "train_L0_p4/p4-4Fs-1D-cnn"
LOG_DIR = "/home/ubuntu/tmp/NSC/codes/src/train/logs"
# === compute one example distance ===


# load the config file
config_path = Path(LOG_DIR) / EXP_ID / "config.yaml"
model_path = Path(LOG_DIR) / EXP_ID / "model" / "best_model.pt"

config = OmegaConf.load(config_path)
config.log_dir = str(Path(LOG_DIR) / EXP_ID)

solver = Solver(config)
solver.init_inference().load_model(config, model_path, "cpu")


# prepare one seqC trained
# load the pretrained model

# prepare one seqC validation


# compute all samples distance
