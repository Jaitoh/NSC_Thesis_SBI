import numpy as np

def seqC_nan2num_norm(seqC, nan2num=-1):
    """ fill the nan of the seqC with nan2num and normalize to (0, 1)
    """
    seqC = np.nan_to_num(seqC, nan=nan2num)
    # normalize the seqC from (nan2num, 1) to (0, 1)
    seqC = (seqC - nan2num) / (1 - nan2num)

    return seqC


def seqC_pattern_summary(seqC, summary_type=1, dur_max=15):

    """ extract the input sequence pattern summary from the input seqC

        can either input a array of shape (N, 15) 
        or a dictionary of pulse sequences contain all the information listed below for the further computation
        
        Args:
            seqC (np.array): input sequence of shape (N, 15)  !should be 2 dimensional
                e.g.  np.array([[0, 0.4, -0.4, 0, 0.4, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                [0, 0.4, -0.4, 0, 0.4, 0.4, -0.4, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])

            summary_type:   (default: 1)
                0: with separate left and right (same/oppo/new)
                1: combine left and right (same/oppo/new)

        Return:
            summary_type 0:
            x0 (np.array): input pattern summary of shape (N, 11)
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
            x1 (np.array): input pattern summary of shape (N, 8)
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
    MS      = np.apply_along_axis(lambda x: np.unique(np.abs(x[(~np.isnan(x))&(x!=0)])), axis=1, arr=seqC).reshape(-1)
    _dur    = np.apply_along_axis(lambda x: np.sum(~np.isnan(x)), axis=1, arr=seqC)
    _nLeft  = np.apply_along_axis(lambda x: np.sum(x<0), axis=1, arr=seqC)
    _nRight = np.apply_along_axis(lambda x: np.sum(x>0), axis=1, arr=seqC)
    _nPulse = _dur - _nLeft - _nRight

    # summary of effect stimulus
    dur     = (_dur-1)/(dur_max-1)
    nLeft   = _nLeft/(dur_max-1)
    nRight  = _nRight/(dur_max-1)
    nPause  = (_dur-1-_nLeft-_nRight)/(dur_max-1)

    # extract internal pattern summary
    hist_nSame  = np.apply_along_axis(lambda x: np.sum(x*np.append(0, x[0:-1])>0), axis=1, arr=seqC)/(_dur-1)
    hist_nLsame = np.apply_along_axis(lambda x: np.sum((x*np.append(0, x[0:-1])>0) & (x<0)), axis=1, arr=seqC)/(_dur-1)
    hist_nRsame = np.apply_along_axis(lambda x: np.sum((x*np.append(0, x[0:-1])>0) & (x>0)), axis=1, arr=seqC)/(_dur-1)

    hist_nOppo  = np.apply_along_axis(lambda x: np.sum(x*np.append(0, x[0:-1])<0), axis=1, arr=seqC)/(_dur-1)
    hist_nLoppo = np.apply_along_axis(lambda x: np.sum((x*np.append(0, x[0:-1])<0) & (x<0)), axis=1, arr=seqC)/(_dur-1)
    hist_nRoppo = np.apply_along_axis(lambda x: np.sum((x*np.append(0, x[0:-1])<0) & (x>0)), axis=1, arr=seqC)/(_dur-1)

    hist_nElse  = np.apply_along_axis(lambda x: np.sum( (x*np.append(0, x[0:-1])==0) & (x!=0) ), axis=1, arr=seqC)/(_dur-1)
    hist_nLelse = np.apply_along_axis(lambda x: np.sum( (x*np.append(0, x[0:-1])==0) & (x<0) ), axis=1, arr=seqC)/(_dur-1)
    hist_nRelse = np.apply_along_axis(lambda x: np.sum( (x*np.append(0, x[0:-1])==0) & (x>0) ), axis=1, arr=seqC)/(_dur-1)

    x0 = np.vstack((MS, dur, nLeft, nRight, nPause, hist_nLsame, hist_nLoppo, hist_nLelse, hist_nRsame, hist_nRoppo, hist_nRelse)).T
    x1 = np.vstack((MS, dur, nLeft, nRight, nPause, hist_nSame, hist_nOppo, hist_nElse)).T

    if summary_type == 0:
        return x0
    else:  # default output
        return x1


def probR_sampling_for_choice(probR, num_probR_sample):
    """ sample the probability of right choice from the input probR

        Args:
            probR (np.array): input probability of right choice of shape (N, 1)
            num_probR_sample (int): number of samples for each input probability of right choice

        Return:
            probR_sample (np.array): sampled probability of right choice of shape (N, num_probR_sample)
    """
    if not isinstance(probR, np.ndarray):
        probR = np.array(probR).reshape(-1, 1)

    choice = np.array([
        np.random.choice([0, 1], size=num_probR_sample, p=[1 - prob[0], prob[0]])
        for prob in probR
    ]).reshape(-1, 1)

    return choice


def probR_threshold_for_choice(probR, threshold):
    """ get right choice from the probR, when probR > threshold, choose right(1) else Left(0)

        Args:
            probR (np.array): input probability of right choice of shape (N, 1)
            threshold (float): threshold for right choice

        Return:
            choice (np.array): sampled probability of right choice of shape (N, num_probR_sample)
    """

    if not isinstance(probR, np.ndarray):
        probR = np.array(probR).reshape(-1, 1)

    choice = np.array([
        1 if prob[0] >= threshold else 0
        for prob in probR
    ]).reshape(-1, 1)

    return choice

if __name__ == '__main__':

    # test seqC_pattern_summary
    # seqC = np.array([[0, 0.4, -0.4, 0, 0.4, 0.4, -0.4, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
    seqC = np.array([[0, 0.2, 0, 0.2, 0, -0.2, 0.2, 0, 0, 0.2, 0, np.nan, np.nan, np.nan, np.nan]])
    x0 = seqC_pattern_summary(seqC, summary_type=0)
    x1 = seqC_pattern_summary(seqC, summary_type=1)
    x0_ref = np.array([[0.2, 10/14, 1/14, 4/14, 5/14, 0, 0, 0.1, 0, 0.1, 3/10]])
    x1_ref = np.array([[0.2, 10/14, 1/14, 4/14, 5/14, 0, 0.1, 0.4]])

    print(f'input seqC: {seqC}')
    print(f'output summary shape{x0.shape}, x0 {x0}')
    print(f'output summary shape{x1.shape}, x1: {x1}')

    assert np.sum(x0-x0_ref) == 0, f'x0 test failure, should be\n{x0_ref}'
    assert np.sum(x1-x1_ref) == 0, f'x1 test failure, should be\n{x1_ref}'

    # test probR_sampling_for_choice
    num_probR_sample = 10
    choice = probR_sampling_for_choice([[0.2, 0.4, 0.5]], num_probR_sample)
    print(f'input probR: {[[0.2, 0.4, 0.5]]}')
    print(f'output choice shape: {choice.shape}')
    print(f'output choice: {choice.reshape(-1)}')
    assert np.product(choice.shape)==30, 'choice shape incorrect'

    num_probR_sample = 1
    choice = probR_sampling_for_choice([[0.2, 0.4, 0.5]], num_probR_sample)
    print(f'input probR: {[[0.2, 0.4, 0.5]]}')
    print(f'output choice shape: {choice.shape}')
    print(f'output choice: {choice.reshape(-1)}')
    assert np.product(choice.shape)==3, 'choice shape incorrect'

    threshold = 0.45
    choice = probR_threshold_for_choice([[0.2, 0.4, 0.5]], threshold=threshold)
    assert np.sum(choice-np.array([[0], [0], [1]])) == 0, 'choice incorrect'
    print(f'input probR: {[[0.2, 0.4, 0.5]]}')
    print(f'output choice using threshold method: {choice.reshape(-1)} with threshold {threshold}')