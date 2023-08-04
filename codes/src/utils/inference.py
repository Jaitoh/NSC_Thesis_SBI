from pathlib import Path
from omegaconf import OmegaConf


# load config
def load_stored_config(exp_dir):
    config_path = Path(exp_dir) / "config.yaml"
    config_path = config_path.expanduser()
    print(f"==>> config_path: {config_path}")
    config = OmegaConf.load(config_path)
    config.log_dir = str(exp_dir)

    model_path = Path(exp_dir) / "model" / "best_model.pt"
    # check if model exists
    if not model_path.exists():
        model_path = Path(exp_dir) / "model" / "model_check_point.pt"

    return config, model_path


def get_posterior(model_path, config, device, Solver, low_batch=0):
    """get the trained posterior"""

    solver = Solver(config, training_mode=False)
    solver.init_inference().prepare_dataset_network(
        config,
        model_path,
        device=device,
        low_batch=low_batch,
    )
    posterior = solver.inference.build_posterior(solver.inference._neural_net)
    solver.inference._model_bank = []
    return solver, posterior
