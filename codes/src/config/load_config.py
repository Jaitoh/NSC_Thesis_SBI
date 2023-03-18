import yaml
from pathlib import Path
import warnings

def load_config(config_simulator_path=None,
                config_dataset_path=None,
                config_train_path=None
                ):
    """

    Args:
        config_simulator_path:
        config_dataset_path:
        config_train_path:

    Returns:

    """
    if config_simulator_path is None:
        config_simulator_path = Path('./src/config') / 'simulator_Ca_Pa_Ma.yaml'
        warnings.warn(f'config_simulator_path is not specified, use default: {config_simulator_path}')
        if not config_simulator_path.exists():
            raise FileNotFoundError(f'config_simulator_path: {config_simulator_path} does not exist')

    if config_dataset_path is None:
        config_dataset_path = Path('./src/config') / 'dataset_Sa_Ra.yaml'
        warnings.warn(f'config_dataset_path is not specified, use default: {config_dataset_path}')
        if not config_dataset_path.exists():
            raise FileNotFoundError(f'config_dataset_path: {config_dataset_path} does not exist')

    if config_train_path is None:
        config_train_path = Path('./src/config') / 'train.yaml'
        warnings.warn(f'config_train_path is not specified, use default: {config_train_path}')
        if not config_train_path.exists():
            raise FileNotFoundError(f'config_train_path: {config_train_path} does not exist')

    with open(config_simulator_path, 'r') as f:
        config_simulator = yaml.load(f, Loader=yaml.FullLoader)

    with open(config_dataset_path, 'r') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    with open(config_train_path, 'r') as f:
        config_train = yaml.load(f, Loader=yaml.FullLoader)

    config = {**config_simulator, **config_data, **config_train}

    return config

if __name__ == '__main__':

    test = False

    if test:
        config = load_config(
            config_simulator_path=Path('./src/config') / 'test_simulator.yaml',
            config_dataset_path=Path('./src/config') / 'test_dataset.yaml',
            config_train_path=Path('./src/config') / 'test_train.yaml',
        )
    else:
        config = load_config(
            config_simulator_path=Path('./src/config') / 'simulator_Ca_Pa_Ma.yaml',
        )

    print(config.keys())

    # load and merge yaml files
    config = load_config()
    print(config.keys())