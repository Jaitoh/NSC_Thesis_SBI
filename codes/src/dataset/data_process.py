import torch
import numpy as np
import time

def reshape_shuffle_x_theta(x, theta):
    
    print(f'\nbefore reshape_shuffle_x_theta: \nx.shape={x.shape}, theta.shape={theta.shape}')
    D, M, S, T, C, L_x = x.shape
    _, _, _, _, _, L_theta = theta.shape
    
    x = torch.tensor(x, dtype=torch.float32)
    theta = torch.tensor(theta, dtype=torch.float32)

    x = x.permute(0, 3, 4, 5, 1, 2).contiguous().view(D, T, C, L_x, M*S)
    x = x.permute(0, 1, 2, 4, 3) # D,T,C,M*S,L_x
    x = x.permute(1, 2, 4, 0, 3).contiguous().view(T, C, L_x, D*M*S) 
    x = x.permute(2, 3, 0, 1).contiguous().view(L_x, D*M*S, T*C)
    x = x.permute(2, 1, 0)

    theta = theta.permute(0, 3, 4, 5, 1, 2).contiguous().view(D, T, C, L_theta, M*S)
    theta = theta.permute(0, 1, 2, 4, 3) # D,T,C,M*S,L_theta
    theta = theta.permute(1, 2, 4, 0, 3).contiguous().view(T, C, L_theta, D*M*S)
    theta = theta.permute(2, 3, 0, 1).contiguous().view(L_theta, D*M*S, T*C)
    theta = theta.permute(2, 1, 0)

    # Select the first L_theta component of each sequence
    theta = theta[:, 0, :]

    x_processed = torch.empty_like(x)
    for tc in range(T * C):
        x_temp = x[tc, :, :]
        idx = torch.randperm(D * M * S)
        x_processed[tc, :, :] = x_temp[idx, :]

    x = x_processed

    print('\nreshaped and shuffled: \nx.shape', x.shape, 'theta.shape', theta.shape)

    return x, theta

def seqC_nan2num_norm(seqC, nan2num=-1):
    """ fill the nan of the seqC with nan2num and normalize to (0, 1)
    """
    seqC = np.nan_to_num(seqC, nan=nan2num)
    # normalize the seqC from (nan2num, 1) to (0, 1)
    seqC = (seqC - nan2num) / (1 - nan2num)

    return seqC

import torch

def seqC_nan2num_norm_torch(seqC, nan2num=-1):
    """Fill the NaNs in seqC with nan2num and normalize to (0, 1) using PyTorch."""
    # assert seqC is a torch tensor
    assert isinstance(seqC, torch.Tensor), "seqC should be a torch tensor given your choice of using PyTorch to process"
    
    # Fill NaNs with nan2num
    seqC[seqC.isnan()] = nan2num

    # Normalize the seqC from (nan2num, 1) to (0, 1)
    seqC = (seqC - nan2num) / (1 - nan2num)

    return seqC

# TODO change into torch
def seqC_pattern_summary(seqC, summary_type=0, dur_max=15):

    """ extract the input sequence pattern summary from the input seqC

        can either input a array of shape (D,M,S,T, 15)
        or a dictionary of pulse sequences contain all the information listed below for the further computation
        
        Args:
            seqC (np.array): input sequence of shape (D,M,S,T, 15)  !should be 2 dimensional
                e.g.  np.array([[0, 0.4, -0.4, 0, 0.4, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                [0, 0.4, -0.4, 0, 0.4, 0.4, -0.4, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])

            summary_type:   (default: 1)
                0: with separate left and right (same/oppo/new) detailed
                1: combine left and right (same/oppo/new) brief

        Return:
            summary_type 0:
            x0 (np.array): input pattern summary of shape (D,M,S,T,C, 11)
                column 1: MS
                column 2: dur
                column 3: nLeft
                column 4: nRight
                column 5: nPulse
                column 6: hist_nLsame
                column 7: hist_nLoppo
                column 8: hist_nLelse
                column 9: hist_nRsame
                column 10: hist_nRoppo
                column 11: hist_nRelse

            summary_type 1:
            x1 (np.array): input pattern summary of shape (D,M,S,T,C, 8)
                column 1: MS
                column 2: dur
                column 3: nLeft
                column 4: nRight
                column 5: nPulse
                column 6: hist_nSame
                column 7: hist_nOppo
                column 8: hist_nElse
                
    """
    
    # get the MS of each trial
    MS      = np.apply_along_axis(lambda x: np.unique(np.abs(x[(~np.isnan(x))&(x!=0)])), axis=-1, arr=seqC)
    MS      = np.reshape(MS, (*seqC.shape[:-1], )) # MS = MS[:,:,:,:,:,-1]
    
    _dur    = np.apply_along_axis(lambda x: np.sum(~np.isnan(x)), axis=-1, arr=seqC)
    _nLeft  = np.apply_along_axis(lambda x: np.sum(x<0), axis=-1, arr=seqC)
    _nRight = np.apply_along_axis(lambda x: np.sum(x>0), axis=-1, arr=seqC)
    _nPulse = _dur - _nLeft - _nRight

    # summary of effect stimulus
    dur     = (_dur-1)/(dur_max-1)
    nLeft   = _nLeft/(dur_max-1)
    nRight  = _nRight/(dur_max-1)
    nPause  = (_dur-1-_nLeft-_nRight)/(dur_max-1)

    # extract internal pattern summary
    hist_nSame  = np.apply_along_axis(lambda x: np.sum(x*np.append(0, x[0:-1])>0), axis=-1, arr=seqC)/(_dur-1)
    hist_nLsame = np.apply_along_axis(lambda x: np.sum((x*np.append(0, x[0:-1])>0) & (x<0)), axis=-1, arr=seqC)/(_dur-1)
    hist_nRsame = np.apply_along_axis(lambda x: np.sum((x*np.append(0, x[0:-1])>0) & (x>0)), axis=-1, arr=seqC)/(_dur-1)

    hist_nOppo  = np.apply_along_axis(lambda x: np.sum(x*np.append(0, x[0:-1])<0), axis=-1, arr=seqC)/(_dur-1)
    hist_nLoppo = np.apply_along_axis(lambda x: np.sum((x*np.append(0, x[0:-1])<0) & (x<0)), axis=-1, arr=seqC)/(_dur-1)
    hist_nRoppo = np.apply_along_axis(lambda x: np.sum((x*np.append(0, x[0:-1])<0) & (x>0)), axis=-1, arr=seqC)/(_dur-1)

    hist_nElse  = np.apply_along_axis(lambda x: np.sum( (x*np.append(0, x[0:-1])==0) & (x!=0) ), axis=-1, arr=seqC)/(_dur-1)
    hist_nLelse = np.apply_along_axis(lambda x: np.sum( (x*np.append(0, x[0:-1])==0) & (x<0) ), axis=-1, arr=seqC)/(_dur-1)
    hist_nRelse = np.apply_along_axis(lambda x: np.sum( (x*np.append(0, x[0:-1])==0) & (x>0) ), axis=-1, arr=seqC)/(_dur-1)

    # add one more dimension for concatenation
    MS          = np.expand_dims(MS, axis=-1)
    dur         = np.expand_dims(dur, axis=-1)
    nLeft       = np.expand_dims(nLeft, axis=-1)
    nRight      = np.expand_dims(nRight, axis=-1)
    nPause      = np.expand_dims(nPause, axis=-1)
    hist_nLsame = np.expand_dims(hist_nLsame, axis=-1)
    hist_nLoppo = np.expand_dims(hist_nLoppo, axis=-1)
    hist_nLelse = np.expand_dims(hist_nLelse, axis=-1)
    hist_nRsame = np.expand_dims(hist_nRsame, axis=-1)
    hist_nRoppo = np.expand_dims(hist_nRoppo, axis=-1)
    hist_nRelse = np.expand_dims(hist_nRelse, axis=-1)
    hist_nSame  = np.expand_dims(hist_nSame, axis=-1)
    hist_nOppo  = np.expand_dims(hist_nOppo, axis=-1)
    hist_nElse  = np.expand_dims(hist_nElse, axis=-1)

    # concatenate the summary along the 5th dimension
    x0 = np.concatenate((MS, dur, nLeft, nRight, nPause, hist_nLsame, hist_nLoppo, hist_nLelse, hist_nRsame, hist_nRoppo, hist_nRelse), axis=-1)
    x1 = np.concatenate((MS, dur, nLeft, nRight, nPause, hist_nSame, hist_nOppo, hist_nElse), axis=-1)

    if summary_type == 0:
        return x0
    else:  # default output
        return x1


def probR_sampling_for_choice(probR, num_probR_sample=10):
    """ sample the probability of right choice from the input probR

        Args:
            probR (np.array): input probability of right choice of shape (D,M,S,T, 1)
            num_probR_sample (int): number of samples for each input probability of right choice

        Return:
            probR_sample (np.array): sampled probability of right choice of shape (D,M,S,T, num_probR_sample(C), 1)
    """
    # if not isinstance(probR, np.ndarray):
    #     probR = np.array(probR)
    # probR = np.reshape(probR, (*probR.shape[:-1], ))
    #
    # choice = np.empty((*probR.shape, num_probR_sample))
    # for D in range(probR.shape[0]):
    #     for M in range(probR.shape[1]):
    #         for S in range(probR.shape[2]):
    #             for T in range(probR.shape[3]):
    #                 prob = probR[D, M, S, T]
    #                 # choice[D, M, S, :] = np.random.choice([0, 1], size=num_probR_sample, p=[1 - prob, prob])
    #                 cs = np.random.binomial(1, prob, size=num_probR_sample)
    #                 choice[D, M, S, T, :] = cs
    # choice = choice[:, :, :, :, :, np.newaxis]
    # # choice.shape = (D,M,S,T, num_probR_sample)
    
    probR_origin = probR
    # torch version TODO compare the speed
    if not isinstance(probR, torch.Tensor):
        probR = torch.tensor(probR) # (D,M,S,T, 1)

    probR  = probR.repeat_interleave(num_probR_sample, dim=-1) # (D,M,S,T, C)
    choice = torch.bernoulli(probR) # (D,M,S,T, C)
    choice = choice.unsqueeze_(dim=-1) # (D,M,S,T, C, 1)
    choice = np.array(choice)

    return choice


def probR_threshold_for_choice(probR, threshold=0.5):
    """ get right choice from the probR, when probR > threshold, choose right(1) else Left(0)

        Args:
            probR (np.array): input probability of right choice of shape (D,M,S,T, 1)
            threshold (float): threshold for right choice

        Return:
            choice (np.array): sampled probability of right choice of shape (D,M,S,T,C, 1)
    """

    if not isinstance(probR, np.ndarray):
        probR = np.array(probR).reshape(-1, 1)
    choice = np.where(probR > threshold, 1, 0)
    choice = choice[:, :, :, :, :, np.newaxis]
    return choice


def process_x_seqC_part(
    seqC, 
    seqC_process,
    nan2num,
    summary_type,
    torch=False,
):

    # input seqC is a 2D array with shape (num_seqC, seqC_len)

    if len(seqC.shape) == 1:
        seqC = seqC.reshape(1, -1)

    
    if seqC_process == 'norm':
        if torch:
            seqC = seqC_nan2num_norm_torch(seqC, nan2num=nan2num)
        else:
            seqC = seqC_nan2num_norm(seqC, nan2num=nan2num)
            
    elif seqC_process == 'summary':
        # TODO add torch version
        seqC = seqC_pattern_summary(seqC, summary_type=summary_type)
        
    else:
        raise ValueError(f'Invalid seqC_process: {seqC_process}')

    return seqC