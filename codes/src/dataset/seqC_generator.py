"""
generate input sequence for model
"""
import numpy as np


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
        self.sample_size = 700

    def generate(self,
                 MS_list=None,
                 dur_max=15,
                 sample_size=700,
                 ):
        """
        single_dur: if 0, generate samples of all dur_length output shape, -> output shape [7, len(MS_list), sample_size, 15]
                    if not 0, generate one sample of length single_dur -> output shape [len(MS_list)*sample_size, single_dur]
        """
        
        if MS_list is None:
            MS_list = [0.2, 0.4, 0.8]

        self.dur_max = dur_max
        if (self.dur_max-self.dur_min) % self.len_stp != 0:
            raise ValueError(f'dur_max setting error, (dur_max-dur_min) % len_stp != 0')

        self.MS_list = MS_list  # motion strength list
        self.sample_size = sample_size  # number of samples to generate for each MS and dur
        
        dur_list = range(self.dur_min, self.dur_max + 1, self.len_stp)
        temp = np.zeros([len(dur_list), len(MS_list), sample_size, dur_max-1])
        for i, dur in enumerate(dur_list):
            for j, MS in enumerate(MS_list):
                temp[i, j, :, :] = self._generate_one(MS, dur)
        zeros = np.zeros([temp.shape[0], temp.shape[1], temp.shape[2], 1])
        temp = np.concatenate([zeros, temp], axis=3)
            
        return temp

    def _generate_one(self, MS, dur) -> np.ndarray: # shape (sample_size, dur_max-1)
        # generate one sample of shape (sample_size, dur_max) with MS and dur
        arr = MS * np.random.choice([-1, 0, 1], size=(self.sample_size, dur-1), p=[self.pL, self.pP, self.pR])

        # avoid the sample with all zeros in a row
        zero_rows = np.where(~arr.any(axis=1))[0]
        while len(zero_rows) > 0:
            arr = np.delete(arr, zero_rows, axis=0)
            new = MS * np.random.choice([-1, 0, 1], size=(len(zero_rows), dur-1), p=[self.pL, self.pP, self.pR])
            arr = np.vstack([arr, new])
            zero_rows = np.where(~arr.any(axis=1))[0]

        return np.pad(arr, ((0, 0), (0, self.dur_max - dur)), 'constant', constant_values=self.nan_padding)


if __name__ == '__main__':
    # test code
    # seqC = seqC_generator(nan_padding=None).generate()
    seqC = seqC_generator(nan_padding=None).generate(single_dur=15)
    print(seqC.shape)

    seqC = seqC_generator(nan_padding=None).generate(single_dur=0)
    print(seqC.shape)