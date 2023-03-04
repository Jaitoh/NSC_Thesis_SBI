from pathlib import Path
import numpy as np
# import pandas as pd
import scipy.io as sio

def parse_trial_data(PathName: str):
    """ parse the trial data from .mat file

    Args:
        PathName (string): the path of the .mat file 
            '../../data/trials.mat'
        
    Returns:
        data (dictionary): trial data of shape (for example):
            pulse:              (214214,15)
            dur:                (214214,1)  duration of each trial
            MS:                 (214214,1)  motion strength
            nLeft:              (214214,1) 
            nRight:             (214214,1)
            nPulse:             (214214,1)
            hist_nLsame:        (214214,1)
            hist_nLoppo:        (214214,1)
            hist_nLelse:        (214214,1)
            hist_nRsame:        (214214,1)
            hist_nRoppo:        (214214,1)
            hist_nRelse:        (214214,1)
            chooseR:            (214214,1)
            subjID:             (214214,1)
            correct:            (214214,1)
    """
    
    filePath = Path(PathName)
    trials = sio.loadmat(filePath)
    
    data = trials['data'][0]
    _info = trials['info'][0]
    info = [_info[i][0] for i in range(len(_info))]

    pulse  = data[0]
    # pulse  = np.nan_to_num(pulse, nan=100)
    dur = data[1]
    MS = data[2]
    nRight = data[3]
    nLeft = data[4]
    nPulse = data[5]
    
    hist_nRelse = data[27]
    hist_nLelse = data[28]
    hist_nRsame = data[29]
    hist_nLsame = data[30]
    hist_nRoppo = data[31]
    hist_nLoppo = data[32]
    
    chooseR = data[-3] #  0: left, 1: Right
    subjID  = data[-1]
    correct = data[-2]

    
    return dict(pulse=pulse, 
                dur=dur,
                MS=MS,
                nRight=nRight,
                nLeft=nLeft,
                nPulse=nPulse,
                hist_nRelse=hist_nRelse,
                hist_nLelse=hist_nLelse,
                hist_nRsame=hist_nRsame,
                hist_nLsame=hist_nLsame,
                hist_nRoppo=hist_nRoppo,
                hist_nLoppo=hist_nLoppo,
                chooseR=chooseR, 
                subjID=subjID, 
                correct=correct, 
                info=info)

def get_unique_seqC_for_subj(subjID, trial_data_dir = '../data/trials.mat'):
    """get the unique sequence of pulse for specified subject

    Args:
        subjID (int): subject ID ranging from 1-15

    Returns:
        pulse (np.array): 
            pulse sequence of shape (nTrials, 15) (with np.nan inside)
        uniquePulse (np.array): 
            unique pulse sequence of shape (nUniqueTrials, 15) (with np.nan inside)
    """
    
    # data = parse_trial_data('../../data/trials.mat')
    data = parse_trial_data(trial_data_dir)
    idx_s1 = np.where(data['subjID']==subjID)[0]
    pulse = data['pulse'][idx_s1]
    pulse = np.nan_to_num(pulse, nan=100)
    uniquePulse = np.unique(pulse, axis=0)

    pulse[pulse==100] = np.nan
    uniquePulse[uniquePulse==100] = np.nan

    return pulse, uniquePulse


def compute_subject_acc(correct_array, in_time=False, time_step=700):
    """compute subject accuracy / correctness in time
    Args:
        correct_array   (np.array):     correctness of trials
        in_time         (bool):         if True, return correctness in time else return correctness in trial
        time_step       (int):          time step 
    """
    
    correct_array = correct_array.copy()
    correctness_all = correctness_of_array(correct_array)
    
    if in_time:
        assert correct_array.shape[0] % time_step == 0, "length of correct_array is not divisible by time_step"
        correct_array = correct_array.reshape(-1,time_step)
        correctness_in_time = []
        for i in range(correct_array.shape[0]):
            correctness_in_time.append(correctness_of_array(correct_array[i]))
        return correctness_in_time, correctness_all
    else:
        return correctness_all
    
    
def correctness_of_array(correct_array):
    return np.mean(correct_array[~np.isnan(correct_array)])

if __name__ == '__main__':
    data = parse_trial_data('../data/trials.mat')
    print(data)
    