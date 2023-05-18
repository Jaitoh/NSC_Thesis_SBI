import cProfile
import pstats
from pathlib import Path
import os
import torch

import sys
sys.path.append('./src')
from train.train_L0 import Solver 
from utils.setup import(
    check_path, get_args,
)
from config.load_config import load_config

def test_train_L0():
    args = get_args(
        config_simulator_path = "./src/config/simulator/exp_set_0.yaml", 
        config_dataset_path = "./src/config/dataset/dataset-config-0.yaml",
        config_train_path = "./src/config/train/train-config-0.yaml",
        data_path = "../data/dataset/dataset_L0_exp_set_0.h5"
    )
    print(args.log_dir)
    
    log_dir = Path(args.log_dir)
    data_path = Path(args.data_path)
    check_path(log_dir, data_path, args)
    
    # monitor resources usage
    PID = os.getpid()
    print(f"PID: {PID}")
    # log_file = f"{args.log_dir}/resource_usage.log"
    # monitor_process = multiprocessing.Process(target=monitor_resources, args=(PID, 5, log_file))
    # monitor_process.start()
    
    try:
        config = load_config(
            config_simulator_path=args.config_simulator_path,
            config_dataset_path=args.config_dataset_path,
            config_train_path=args.config_train_path,
        )

        print(f'\n--- args ---')
        for arg, value in vars(args).items():
            print(f'{arg}: {value}')

        print('\n--- config keys ---')
        print(config.keys())

        solver = Solver(args, config)
        solver.sbi_train(debug=True)
    
    # except Exception as e:
    #         print(f"An error occurred: {e}")
    finally:
        
        torch.cuda.empty_cache()
        print('cuda cache emptied')
        # del solver
        # print('solver deleted')
        

cProfile.run('test_train_L0()', 'output.dat')  # Replace with the name of your Python function
profiler = cProfile.Profile()
profiler.enable()
test_train_L0()
profiler.disable()
# stats = pstats.Stats(profiler).sort_stats('tottime')
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats()
stats.dump_stats('./cProfile_train_L0.txt')
