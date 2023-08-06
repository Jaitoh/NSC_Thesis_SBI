from pathlib import Path
import numpy as np

# import pandas as pd
import torch
import scipy.io as sio

import sys
from pathlib import Path

NSC_DIR = Path(__file__).resolve().parent.parent.parent.parent.as_posix()  # NSC dir
sys.path.append(f"{NSC_DIR}/codes/src")

from dataset.data_process import process_x_seqC_part
from utils.setup import adapt_path


def get_xo(
    data_path,
    subj_ID=2,
    dur_list=[3],
    MS_list=[0.2],
):
    """get the data from the dataset for a given
    - subject ID,
    - duration,
    - motion strength,
    """
    # data_path = "/home/wehe/tmp/NSC/data/trials.mat"
    data_subjects = parse_trial_data(data_path)

    seqCs = data_subjects["pulse"]
    chRs = data_subjects["chooseR"]

    # check element in durs in dur_list, of subject and return the index
    subj_IDs = data_subjects["subjID"]
    durs = data_subjects["dur"]
    MSs = data_subjects["MS"]
    idx_dur = np.isin(durs, dur_list)
    idx_subj = subj_IDs == subj_ID
    idx_MS = np.isin(MSs, MS_list)
    idx_chosen = idx_dur & idx_subj & idx_MS
    print("".center(50, "-"))
    print(f"{sum(idx_chosen)} samples are chosen with subj_ID={subj_ID}, dur={dur_list}, MS={MS_list}")
    print("".center(50, "-"))

    idx_chosen = idx_chosen.reshape(-1)
    seqC = seqCs[idx_chosen, :]
    chR = chRs[idx_chosen, :]

    # process the data, and not ignore the first element in seqC
    seqC = process_x_seqC_part(seqC)[:, :]

    return torch.from_numpy(seqC), torch.from_numpy(chR)


def parse_trial_data(PathName: str):
    """parse the trial data from .mat file

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

    filePath = adapt_path(PathName)
    trials = sio.loadmat(filePath)

    data = trials["data"][0]
    _info = trials["info"][0]
    info = [_info[i][0] for i in range(len(_info))]

    pulse = data[0]
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

    chooseR = data[-3]  #  0: left, 1: Right
    subjID = data[-1]
    correct = data[-2]

    return dict(
        pulse=pulse,
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
        info=info,
    )


def decode_mat_fitted_parameters(filePath):
    """load fitted parameters from .mat file

    Args:
        filePath (string): the path of the .mat file
            e.g. '../../data/params/263 models fitPars/data_fitPars_S1.mat'

    Returns:
        params (dictionary): fitted parameters of shape (for example):
            bias:               (263,)
            sigmas:             (263, 3)
            BGLS:               (263, 6, 8) - 7 is the highest order of orthogonal polynomials, 8 is the highest number of parameters
            mechanismsMatrix:   (263, 6)
            probRchoiceStoch:   (263, 14994)
            allModelsList:      (263,)
    """

    dataFitPars = sio.loadmat(filePath)["dataFitPars"][0][0]

    bias = dataFitPars[0]
    sigmas = dataFitPars[1]
    BGLS = dataFitPars[2][0]
    mechanismsMatrix = dataFitPars[3]
    probRchoiceStoch = dataFitPars[4]
    allModelsList = dataFitPars[5]

    params = {}
    params["bias"] = bias[0]  # reshape bias to (263,)
    params["sigmas"] = sigmas  # reshape sigmas to (263, 6)

    # reshape BGLS to (263, 6, 8)
    ps = []
    for i in range(len(BGLS)):
        _, num_col = BGLS[i].shape
        if num_col < 8:
            p_temp = np.pad(BGLS[i], [(0, 0), (0, 8 - num_col)], mode="constant", constant_values=np.nan)
        else:
            p_temp = BGLS[i][:, :8]
        ps.append(p_temp)
    params["BGLS"] = np.array(ps)

    params["mechanismsMatrix"] = mechanismsMatrix.T  # reshape mechanismsMatrix to (263, 6)
    params["probRchoiceStoch"] = np.array(
        [probRchoiceStoch[0][i][0] for i in range(len(probRchoiceStoch[0]))]
    )  # extract and reshape probRchoiceStoch to (263,14994)
    params["allModelsList"] = [
        allModelsList[i][0][0] for i in range(len(allModelsList))
    ]  # reshape allModelsList to (263,)

    return params


def get_unique_seqC_for_subj(subjID, trial_data_dir="../data/trials.mat"):
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
    idx_s1 = np.where(data["subjID"] == subjID)[0]
    pulse = data["pulse"][idx_s1]
    pulse = np.nan_to_num(pulse, nan=100)
    uniquePulse = np.unique(pulse, axis=0)

    pulse[pulse == 100] = np.nan
    uniquePulse[uniquePulse == 100] = np.nan

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
        assert (
            correct_array.shape[0] % time_step == 0
        ), "length of correct_array is not divisible by time_step"
        correct_array = correct_array.reshape(-1, time_step)
        correctness_in_time = []
        for i in range(correct_array.shape[0]):
            correctness_in_time.append(correctness_of_array(correct_array[i]))
        return correctness_in_time, correctness_all
    else:
        return correctness_all


def correctness_of_array(correct_array):
    return np.mean(correct_array[~np.isnan(correct_array)])


if __name__ == "__main__":
    data = parse_trial_data("../data/trials.mat")
    print(data)
