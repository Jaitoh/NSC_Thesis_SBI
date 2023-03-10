"""
generate input sequence for model
"""
import numpy as np


class seqCGenerator:
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
                 single_dur=0,
                 # add_zero=True,
                 ):

        if MS_list is None:
            MS_list = [0.2, 0.4, 0.8]

        self.dur_max = dur_max
        if (self.dur_max-self.dur_min) % self.len_stp != 0:
            raise ValueError(f'dur_max setting error, (dur_max-dur_min) % len_stp != 0')

        self.MS_list = MS_list  # motion strength list
        self.sample_size = sample_size  # number of samples to generate for each MS and dur

        if single_dur == 0:  # generate samples of different dur_length

            temp = np.zeros([1, self.dur_max-1])
            for dur in range(self.dur_min, self.dur_max + 1, self.len_stp):
                for MS in self.MS_list:
                    temp = np.vstack([temp, self._generate_one(MS, dur)])
            temp = temp[1:, :]

        else:  # generate one sample of length single_dur

            temp = np.empty([self.sample_size * len(MS_list), self.dur_max-1])
            dur = single_dur
            for i, MS in enumerate(self.MS_list):
                temp[i * self.sample_size:(i + 1) * self.sample_size, :] = self._generate_one(MS, dur)

        # if add_zero:
        zeros = np.zeros([temp.shape[0], 1])
        temp = np.hstack([zeros, temp])

        return temp

    def _generate_one(self, MS, dur):
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

    def set_dur_max(self, dur_max):
        self.dur_max = dur_max


if __name__ == '__main__':
    # test code
    # seqC = seqCGenerator(nan_padding=None).generate()
    seqC = seqCGenerator(nan_padding=None).generate(single_dur=15)
    print(seqC.shape)
