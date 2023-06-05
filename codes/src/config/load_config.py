import yaml
from pathlib import Path
import warnings
import hydra
from omegaconf import DictConfig
import os
from hydra.utils import get_original_cwd, to_absolute_path

def parse_dataset_path(config_dataset_path):
    """

    Args:
        config_dataset_path: in the format of './src/config/dataset_Sa0_suba0_Ra0.yaml'

    Returns:

    """
    config_dataset_path = str(config_dataset_path)
    config_dataset_dir = config_dataset_path.split('/')[:-1]
    config_dataset_dir = '/'.join(config_dataset_dir) + '/dataset'
    
    data_abbrs = config_dataset_path.split('/')[-1].split('.')[0].split('_')[1:]
    data_files = [f'dataset_{ab}.yaml' for ab in data_abbrs]
    
    config_dataset_paths= [config_dataset_dir + '/' + f for f in data_files]
    dataset_abbr = '_'.join(data_abbrs)
    
    return config_dataset_paths, dataset_abbr


def load_configs_dataset(
    config_dataset_paths, dataset_abbr
):
    config_collection = {}
    # load config files and merge them
    for i, config_path in enumerate(config_dataset_paths):
        config = load_1config(config_path)
        config_collection.update(config)
    dataset_config = {'train_data': config_collection}

    save_name = f'train_data_{dataset_abbr}.h5'
    dataset_config['train_data']['save_name'] = save_name

    return dataset_config

def load_1config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f, Loader=yaml.FullLoader)  # type: ignore
    return config

def load_config(config_simulator_path=None,
                config_dataset_path=None,
                config_train_path=None,
                ):
    """

    Args:
        config_simulator_path:
        config_dataset_path:
        config_train_path:

    Returns:
        config
    """
    if config_simulator_path is None:
        config_simulator_path = Path('./src/config') / 'test' / 'test_simulator.yaml'
        warnings.warn(f'config_simulator_path is not specified, use default: {config_simulator_path}')
        if not config_simulator_path.exists():
            raise FileNotFoundError(f'config_simulator_path: {config_simulator_path} does not exist')

    if config_train_path is None:
        config_train_path = Path('./src/config') / 'test'/ 'test_train.yaml'
        warnings.warn(f'config_train_path is not specified, use default: {config_train_path}')
        if not config_train_path.exists():
            raise FileNotFoundError(f'config_train_path: {config_train_path} does not exist')


    with open(config_simulator_path, 'r') as f:
        config_simulator = yaml.load(f, Loader=yaml.FullLoader)

    with open(config_train_path, 'r') as f:
        config_train = yaml.load(f, Loader=yaml.FullLoader)


    if config_dataset_path is None:
        config_dataset_path = Path('./src/config') / 'test' /'test_dataset.yaml'
        warnings.warn(f'config_dataset_path is not specified, use default: {config_dataset_path}')

    # check if the file exists
    if Path(config_dataset_path).exists():
        with open(config_dataset_path, 'r') as f:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
    else:
        # parse the dataset path into a list of files
        config_dataset_paths, dataset_abbr = parse_dataset_path(config_dataset_path)
        config_data = load_configs_dataset(config_dataset_paths, dataset_abbr)

    config = {**config_simulator, **config_data, **config_train}

    return config
    
if __name__ == '__main__':

    # config = load_config(
    #     config_simulator_path=Path('./src/config') / 'test' / 'test_simulator.yaml',
    #     config_dataset_path=Path('./src/config') / 'test' /'test_dataset.yaml',
    #     config_train_path=Path('./src/config') / 'test' / 'test_train.yaml',
    # )
    # print(config.keys())
    # print(config)

    # config = load_config(
    #     config_simulator_path=Path('./src/config') / 'simulator' / 'exp-set-0.yaml',
    #     config_dataset_path=Path('./src/config') / 'dataset' / 'dataset-p2-0.yaml',
    #     config_train_path=Path('./src/config') / 'train' / 'train-p2-test-0.yaml',
    # )

    # print(config.keys())
    # print(config)

    # load and merge yaml files
    config = load_config()
    print(config.keys())
    print(config)
    
    