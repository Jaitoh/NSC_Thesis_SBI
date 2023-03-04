import scipy.io as sio
import numpy as np

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
    
    dataFitPars = sio.loadmat(filePath)['dataFitPars'][0][0]

    bias   = dataFitPars[0]
    sigmas = dataFitPars[1]
    BGLS   = dataFitPars[2][0]
    mechanismsMatrix = dataFitPars[3]
    probRchoiceStoch = dataFitPars[4]
    allModelsList    = dataFitPars[5]
    
    params = {}
    params['bias'] = bias[0] # reshape bias to (263,)
    params['sigmas'] = sigmas # reshape sigmas to (263, 6)
    
    # reshape BGLS to (263, 6, 8)
    ps = []
    for i in range(len(BGLS)):
        _, num_col = BGLS[i].shape
        if num_col < 8:
            p_temp = np.pad(BGLS[i], [(0, 0), (0, 8 - num_col)], mode='constant', constant_values=np.nan)
        else: 
            p_temp = BGLS[i][:, :8]
        ps.append(p_temp)
    params['BGLS'] = np.array(ps)
    
    params['mechanismsMatrix'] = mechanismsMatrix.T # reshape mechanismsMatrix to (263, 6)
    params['probRchoiceStoch'] = np.array([probRchoiceStoch[0][i][0] for i in range(len(probRchoiceStoch[0]))]) # extract and reshape probRchoiceStoch to (263,14994)
    params['allModelsList'] = [allModelsList[i][0][0] for i in range(len(allModelsList))] # reshape allModelsList to (263,)
    
    return params
