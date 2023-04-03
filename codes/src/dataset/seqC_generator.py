"""
generate input sequence for model
"""
import sys
sys.path.append('./src')
import numpy as np
from config.load_config import load_config
from pathlib import Path

class seqC_generator:
    def __init__(self, nan_padding=None):

        self.dur_min = 3
        self.dur_max = 15
        self.len_stp = 2

        self.pR = 0.35  # probability of rightward motion
        self.pL = 0.35  # probability of leftward motion
        self.pP = 0.3  # probability of pause

        self.nan_padding = np.nan if nan_padding is None else nan_padding

        self.MS_list = [0.2, 0.4, 0.8]
        self.seqC_sample_per_MS = 700

    def generate(self,
                 dur_list=None,
                 MS_list=None,
                 seqC_sample_per_MS=700,
                 ):
        """
        generate input sequence for model
        """
        
        if MS_list is None:
            MS_list = [0.2, 0.4, 0.8]
        if dur_list is None:
            dur_list = [3,5,7,9,11,13,15]
        
        self.MS_list = MS_list  # motion strength list
        self.seqC_sample_per_MS = seqC_sample_per_MS  # number of samples to generate for each MS and dur
        
        temp = np.zeros([len(dur_list), len(MS_list), seqC_sample_per_MS, self.dur_max-1])
        for i, dur in enumerate(dur_list):
            for j, MS in enumerate(MS_list):
                temp[i, j, :, :] = self._generate_one(MS, dur)
        zeros = np.zeros([temp.shape[0], temp.shape[1], temp.shape[2], 1])
        temp = np.concatenate([zeros, temp], axis=-1)
        
        print('---\ngenerated seqC info')
        print('dur_list:', dur_list)
        print('MS_list:', MS_list)
        print('seqC_sample_per_MS:', seqC_sample_per_MS)
        print('generated seqC shape:', temp.shape)
        
        return temp

    def _generate_one(self, MS, dur) -> np.ndarray: # shape (sample_size, dur_max-1)
        # generate one sample of shape (sample_size, dur_max) with MS and dur
        arr = MS * np.random.choice([-1, 0, 1], size=(self.seqC_sample_per_MS, dur-1), p=[self.pL, self.pP, self.pR])

        # avoid the sample with all zeros in a row
        zero_rows = np.where(~arr.any(axis=1))[0]
        while len(zero_rows) > 0:
            arr = np.delete(arr, zero_rows, axis=0)
            new = MS * np.random.choice([-1, 0, 1], size=(len(zero_rows), dur-1), p=[self.pL, self.pP, self.pR])
            arr = np.vstack([arr, new])
            zero_rows = np.where(~arr.any(axis=1))[0]

        return np.pad(arr, ((0, 0), (0, self.dur_max - dur)), 'constant', constant_values=self.nan_padding)


if __name__ == '__main__':
    
    config = load_config(
        config_simulator_path=Path('./src/config') / 'test' / 'test_simulator.yaml',
        config_dataset_path=Path('./src/config') / 'test' / 'test_dataset.yaml',
        config_train_path=Path('./src/config') / 'test' / 'test_train.yaml',
    )
    
    seqC = seqC_generator(nan_padding=None).generate(
        dur_list = config['x_o']['chosen_dur_list'],
        MS_list=config['x_o']['chosen_MS_list'],
        seqC_sample_per_MS=config['x_o']['seqC_sample_per_MS'],
    )
    