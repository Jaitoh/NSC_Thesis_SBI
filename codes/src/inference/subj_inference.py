"""inference with subject data

load posterior
load and process trial data
inference with trial data
"""

from pathlib import Path
import sys

NSC_DIR = Path(__file__).resolve().parent.parent.parent.parent.as_posix()  # NSC dir
sys.path.append(f"{NSC_DIR}/codes/src")

from utils.subject import get_xo
from utils.inference import load_stored_config, get_posterior

exp_dir = "~/tmp/NSC/codes/src/train/logs/train_L0_p5a/p5a-conv_net-old_net"

# load config
config, model_path = load_stored_config(exp_dir=exp_dir)
if "p4" in config.pipeline_version:
    from train.train_L0_p4a import Solver
if "p5" in config.pipeline_version:
    from train.train_L0_p5a import Solver

# load posterior
solver, posterior = get_posterior(
    model_path=model_path,
    config=config,
    device="cpu",
    Solver=Solver,
    low_batch=10,
)

# load and process trial data
data_path = Path(NSC_DIR / "data/trials.mat")
get_xo(data_path)
