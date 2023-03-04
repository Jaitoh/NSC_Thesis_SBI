import numpy as np

def seqC_pattern_summary(*arg, **kwargs):
    
    """ extract the input sequence pattern summary from the input seqC

        can either input a array of shape (N, 15) 
        or a dictionary of pulse sequences contain all the information listed below for the further computation
        
        Args:
            summaryTyepe:   (default: 2)
                            1: with separate left and right (same/oppo/new)
                            2: combine left and right (same/oppo/new)
            *arg (np.array): input sequence of shape (N, 15)  !should be 2 dimensional
                e.g.  np.array([[0, 0.4, -0.4, 0, 0.4, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                [0, 0.4, -0.4, 0, 0.4, 0.4, -0.4, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
            
            **kwargs (dict): the dictionary of pulse sequences
                contain the following keys:
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
        
        Return:
            x1 (np.array): input pattern summary of shape (N, 11)
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
                
            x2 (np.array): input pattern summary of shape (N, 8)
                column 1: MS
                column 2: dur
                column 3: nLeft
                column 4: nRight
                column 5: nPulse
                column 6: hist_nSame
                column 7: hist_nOppo
                column 8: hist_nElse
                
    """
    
    # check if *arg is empty
    if not arg:
        if not kwargs:
            raise ValueError('No input sequence is provided')
        else:
            # process pre-processed data
            _dur = kwargs['dur']
            _nLeft = kwargs['nLeft']
            _nRight = kwargs['nRight']

            dur = (kwargs['dur'] -1) / 14
            nLeft = kwargs['nLeft'] / 14
            nRight = kwargs['nRight'] / 14
            nPulse = (_dur-_nLeft-_nRight-1) / 14
            hist_nLsame = kwargs['hist_nLsame']/(_dur-1)
            hist_nLoppo = kwargs['hist_nLoppo']/(_dur-1)
            hist_nLelse = kwargs['hist_nLelse']/(_dur-1)
            hist_nRsame = kwargs['hist_nRsame']/(_dur-1)
            hist_nRoppo = kwargs['hist_nRoppo']/(_dur-1)
            hist_nRelse = kwargs['hist_nRelse']/(_dur-1)

            hist_nSame = hist_nLsame + hist_nRsame
            hist_nOppo = hist_nLoppo + hist_nRoppo
            hist_nElse = hist_nLelse + hist_nRelse

            x1 = np.hstack((kwargs['MS'], dur, nLeft, nRight, nPulse, hist_nLsame, hist_nLoppo, hist_nLelse, hist_nRsame, hist_nRoppo, hist_nRelse))
            x2 = np.hstack((kwargs['MS'], dur, nLeft, nRight, nPulse, hist_nSame, hist_nOppo, hist_nElse))
            
            if ('summaryType' in kwargs) and kwargs['summaryType'] == 1:
                return x1
            else:
                return x2
            
    else: # given input sequence
        seqC = arg[0]
        
        # get the MS of each trial
        MS      = np.apply_along_axis(lambda x: np.unique(np.abs(x[(~np.isnan(x))&(x!=0)])), axis=1, arr=seqC).reshape(-1)
        _dur    = np.apply_along_axis(lambda x: np.sum(~np.isnan(x)), axis=1, arr=seqC)
        _nLeft  = np.apply_along_axis(lambda x: np.sum(x<0), axis=1, arr=seqC)
        _nRight = np.apply_along_axis(lambda x: np.sum(x>0), axis=1, arr=seqC)
        _nPulse = _dur - _nLeft - _nRight

        hist_nSame  = np.apply_along_axis(lambda x: np.sum(x*np.append(0, x[0:-1])>0), axis=1, arr=seqC)/(_dur-1)
        hist_nLsame = np.apply_along_axis(lambda x: np.sum((x*np.append(0, x[0:-1])>0) & (x<0)), axis=1, arr=seqC)/(_dur-1)
        hist_nRsame = np.apply_along_axis(lambda x: np.sum((x*np.append(0, x[0:-1])>0) & (x>0)), axis=1, arr=seqC)/(_dur-1)

        hist_nOppo  = np.apply_along_axis(lambda x: np.sum(x*np.append(0, x[0:-1])<0), axis=1, arr=seqC)/(_dur-1)
        hist_nLoppo = np.apply_along_axis(lambda x: np.sum((x*np.append(0, x[0:-1])<0) & (x<0)), axis=1, arr=seqC)/(_dur-1)
        hist_nRoppo = np.apply_along_axis(lambda x: np.sum((x*np.append(0, x[0:-1])<0) & (x>0)), axis=1, arr=seqC)/(_dur-1)

        hist_nElse  = np.apply_along_axis(lambda x: np.sum( (x*np.append(0, x[0:-1])==0) & (x!=0) ), axis=1, arr=seqC)/(_dur-1)
        hist_nLelse = np.apply_along_axis(lambda x: np.sum( (x*np.append(0, x[0:-1])==0) & (x<0) ), axis=1, arr=seqC)/(_dur-1)
        hist_nRelse = np.apply_along_axis(lambda x: np.sum( (x*np.append(0, x[0:-1])==0) & (x>0) ), axis=1, arr=seqC)/(_dur-1)

        dur     = (_dur-1)/14
        nLeft   = _nLeft/14
        nRight  = _nRight/14
        nPulse  = (_dur-1-_nLeft-_nRight)/14
        
        x1 = np.vstack((MS, dur, nLeft, nRight, nPulse, hist_nLsame, hist_nLoppo, hist_nLelse, hist_nRsame, hist_nRoppo, hist_nRelse)).T
        x2 = np.vstack((MS, dur, nLeft, nRight, nPulse, hist_nSame, hist_nOppo, hist_nElse)).T
        
        if ('summaryType' in kwargs) and kwargs['summaryType'] == 1:
            return x1
        else: # default output
            return x2
            