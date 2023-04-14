import re
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
import sys
sys.path.append('./src')
# print current working directory
# import os
# print(os.getcwd())
# import numba
import simulator.DM_compute as cython_compute
import matplotlib.pyplot as plt

class DM_model:
    def __init__(self, params, cython=True, **kwargs):
        """ stochastic computing model for decision making

        Args:
            params (dictionary or an array):
                1. dictionary case: 
                    bias:   (1)
                    sigmas: (3)
                    BGLS:   (6, 8) - 7 is the highest order of orthogonal polynomials, 
                                    8 is the highest number of parameters
                2. ararry case: **kwargs['model_name'] is required
                    e.g. [0.5, 0.1, 0, 1, 3], 'bsssB-G-L0S-O-N-' or 'B-G-L0S-O-N-' (bias, sigma2a, sigma2i, sigma2s, L0)
            
            cython (bool): use cython to accelerate the computation
            
            model_name (string): e.g. 'B1G-L0S-O-N-', 'B-G-L0S-O-N-'...
            
        """
        
        self.maxClength             = 15
        self.key_lambdaPolynomial   = 1
        self.temporalDiscretization = 100
        self.numBruns               = 100
        self.dt                     = 11/85 # 11 frame / 85 Hz

        # can call self.simulate() or self.stoch_simualation() to simulate the model
        self.simulate = self.stoch_simulation
        
        self.decode_BGLSON_from_array          = self._decode_BGLSON_from_array
        self.compute_stoch_trace               = self._compute_stoch_trace
        self.compute_mean_trace                = self._compute_mean_trace
        self.affine_transformation             = self._affine_transformation
        self.get_orthogonal_polynomials_curves = self._get_orthogonal_polynomials_curves
        self.sensory_adaptation_for_seqC       = self._sensory_adaptation_for_seqC
        self.compute_variance                  = self._compute_variance
        
        # accelerate the computation by using cython
        if cython:
            # pass
            self.compute_mean_trace                = self._compute_mean_trace_fast
            self.compute_stoch_trace               = self._compute_stoch_trace_fast
            self.compute_variance                  = self._compute_variance_fast
            # self.get_orthogonal_polynomials_curves = self._get_orthogonal_polynomials_curves_fast
            # self.sensory_adaptation_for_seqC       = self._sensory_adaptation_for_seqC_fast
        
        if isinstance(params, dict): # input a dictionary
            # parse parameters
            raise NotImplementedError('dictionary case is not implemented yet')
            # self.params = params
            # self.bias = self.params['bias']
            # self.sigmas  = self.params['sigmas']
            # self.sigmas[self.sigmas<0] = 0 # make sure that sigmas are positive
            # # self.sigmas = 
            # self.sigma2a = self.sigmas[0]
            # self.sigma2i = self.sigmas[1]
            # self.sigma2s = self.sigmas[2]
            # self.BGLSON = self.params['BGLS']
            # self.B = self.params['BGLS'][0,:]
            # self.G = self.params['BGLS'][1,:]
            # self.L = self.params['BGLS'][2,:]
            # self.Ssame = self.params['BGLS'][3,:]
            # self.Soppo = self.params['BGLS'][4,:]
            # self.Selse = self.params['BGLS'][5,:]
            # self.model_name = self.params['model_name']
        
        else:
            
            self.params = np.array(params)
            
            model_name = kwargs['model_name'].upper()
            self.model_name = model_name
            
            BGLSON = self.decode_BGLSON_from_array(self.params, self.model_name)
            self.BGLSON = BGLSON
            self.bias   = self.params[0]
            
            self.sigmas = self.params[1:3]
            
            self.sigmas[self.sigmas<0] = 0 # make sure that sigmas are positive
            self.sigma2a = self.sigmas[0]
            self.sigma2i = 0
            # self.sigma2i = self.sigmas[1]
            self.sigma2s = self.sigmas[1]
            
            self.B = BGLSON[0,:]
            self.G = BGLSON[1,:]
            self.L = BGLSON[2,:]
            self.Ssame = BGLSON[3,:]
            self.Soppo = BGLSON[4,:]
            self.Selse = BGLSON[5,:]
    
    def _decode_BGLSON_from_array(self, params, model_name):

        nan_mat = np.full((6, 8), np.nan)

        params_temp = params[3:]

        # parse the model name
        ns = re.search(r'B(.)G(.)L(.)S(.)O(.)N(.)', model_name).groups()
        _sum = sum([int(x)+1 for x in ns if x != '-'])

        assert _sum + 3 == len(params), 'input parameters dimension do not match the model name'

        for i, n in enumerate(ns):
            if n != '-':
                nan_mat[i, :int(n)+1] = params_temp[:int(n)+1]
                params_temp = params_temp[int(n)+1:]

        return nan_mat
    
    # def _decode_BGLSON_from_array_fast(self, params, model_name):
    #     return cython_compute._decode_BGLSON_from_array_cy(params, model_name)
    def stoch_simulation_2D(self, seqC, debug=False):
        raise NotImplementedError('stoch_simulation_2D is not implemented yet')
        
    
    def stoch_simulation(self, seqC, debug=False):
        """ run the stochastic simulation for a given sequence of C

        Args:
            seqC (array): (lenC, ) input visual stimulus sequence of 1 dimension
            debug (bool): (1,    ) if True, fix the random values for debugging

        Returns:
            a:                  (1, len(seqC)),  the activity of the accumulator
            probRchoiceStoch:   (1, ),          the probability of choosing the right option
        """
        if debug: np.random.seed(111)
        
        
        if not isinstance(seqC, np.ndarray): # convert seqC to numpy array if it's not
            seqC = np.array(seqC) 
        # if seqC.ndim == 1: # if seqC is 1D, then convert it to 2D
        #     seqC = seqC[np.newaxis, :]
            
        seqC = seqC[~np.isnan(seqC)] # remove the nan from seqC
        lenC = len(seqC)
        # if Ssame is not all nan, then do sensory adaptation
        if not np.isnan(self.Ssame).all(): 
            scaleC = np.max(np.abs(seqC))
            if scaleC != 0:
                seqC = seqC / scaleC # making sure that C is in [0;1] C would be  [0,1,-1...]
                seqC = self.sensory_adaptation_for_seqC(seqC)
                seqC = seqC * scaleC
                
        
        B = self.B[~np.isnan(self.B)] # remove nan of parameter B
        G = self.G[~np.isnan(self.G)] # remove nan of parameter G
        L = self.L[~np.isnan(self.L)] # remove nan of parameter L
        
        lenB, lenG, lenL = len(B), len(G), len(L)
        
        order = np.max([lenB, lenG, lenL]) - 1 # highest order of the polynomial
        
        tempBins = np.arange(0, self.dt*(self.maxClength-1), self.dt) # in matlab, the code was tempBins = 0:dt:dt*(maxClength-2) including the maxClength-2 element 0-13dt
        orthogonalPolynomials = self.get_orthogonal_polynomials_curves(tempBins, order)
        
        orthogonalPolynomials = orthogonalPolynomials[:, :lenC-1]# remove the first column of orthogonalPolynomials 
                                                                # no value for 1st pulse (since it's always 0)
        
        # paramTemp = np.ones((3, lenC))
        BTemp = np.inf * np.ones((lenC))   
        GTemp = np.ones((lenC)) 
        LTemp = np.zeros((lenC))          
        
        if lenB > 0:
            BTemp[1:] =  B @ orthogonalPolynomials[:lenB, :]
        if lenG > 0:
            GTemp[1:] = G @ orthogonalPolynomials[:lenG, :]
        if lenL > 0:
            LTemp[1:] = L @ orthogonalPolynomials[:lenL, :]
        
        B           = np.repeat(BTemp, self.temporalDiscretization) # of shape (lenC*temporalDiscretization,)
        GTempDisc   = np.repeat(GTemp, self.temporalDiscretization)
        Lambda      = np.repeat(LTemp, self.temporalDiscretization)
        seqCDisc    = np.repeat(seqC,  self.temporalDiscretization) # of shape (lenC*temporalDiscretization,)
        seqC        = seqCDisc * GTempDisc # input sequence multiplied by the gain
        
        self.aFinal = np.zeros((self.numBruns, len(seqC)+1))
        dt = self.dt/self.temporalDiscretization # time step
        
        # === simulation starts === 
        # deterministic case, compute mean
        if np.sum(self.sigmas) == 0:
            
            a = self.compute_mean_trace(seqC, B, Lambda, dt)
            
            if a[-1] == self.bias:
                probRchoice = 0.5
            else:
                probRchoice = int(a[-1] > self.bias)
                
        # stochastic cases
        else:
            
            if not np.all(np.isinf(B)): # with (sticky) boundary condition
                
                for idxBruns in range(self.numBruns):
                    
                    # print(idxBruns)
                    a = self.compute_stoch_trace(seqC, B, Lambda, dt, debug=debug)
                    self.aFinal[idxBruns, :] = a
                
                probRchoice = np.sum(self.aFinal[:, -1] > self.bias)/self.numBruns
                
                # compute mean once to get the trace
                a = self.compute_mean_trace(seqC, B, Lambda, dt)
                
            else: # no boundary condition
                
                # compute mean once to get the trace
                a = self.compute_mean_trace(seqC, B, Lambda, dt)
                
                sigma2dt = self.compute_variance(seqC, dt, Lambda)
                # sigma2dt_last = sigma2dt[-1] if sigma2dt[-1] != 0 else 1e-6
                sigma2dt_last = sigma2dt[-1]
                probRchoice = 1 - stats.norm.cdf(self.bias, a[-1], np.sqrt(sigma2dt_last))
                cDist = np.random.choice([0,1], p=[1-probRchoice, probRchoice], size=10) 
                assert probRchoice >= 0 and probRchoice <= 1, 'probRchoice should be between 0 and 1'
                assert (probRchoice != np.nan), f'probRchoice is nan with bias={self.bias}, a[-1]={a[-1]}, sigma2dt[-1]={sigma2dt[-1]}'
        # === simulation ends ===

        # self.a = a
        # self.probRchoice = probRchoice

        return a, probRchoice

    def plot_a_mean_trace(self, ax, a, color='tab:blue'):

        # ax = plt.subplot()
        # fig.suptitle('Model: ' + paramsFitted['allModelsList'][idx])
        ax.plot(a[::100], '.-', lw=2, color=color)

        ax.set_xlabel('Time (sample)')
        ax.set_ylabel('a')
        ax.grid(alpha=0.3)

        # set the legend font to bold
        # lgd = plt.legend(loc = 'lower right', fontsize=24)
        # for text in lgd.get_texts():
        #     text.set_fontweight('bold')
        # lgd.get_frame().set_facecolor('none')

        return ax

    def _compute_variance(self, seqC, dt, Lambda):
        
        T = len(seqC)
        
        # compute variance
        var_a    = np.zeros((T))
        var_a[0] = self.sigma2a * dt
        for k in range(1, T):
            var_a[k] = var_a[k-1]*(1+Lambda[k-1]*dt)**2 + self.sigma2a*dt
        
        # Variance of initial noise accumulating with time
        var_i    = np.zeros((T+1))
        var_i[0] = self.sigma2i # at t0
        for k in range(1, T+1):
            var_i[k] = var_i[k-1]*(1 + Lambda[k-1]*dt)**2
        var_i = var_i[1:] # remove the first element
        
        # Variance of sensory noise accumulating with time
        var_s    = np.zeros((T))
        var_s[0] = seqC[0]**2 * dt * self.sigma2s
        for k in range(1, T):   
            var_s[k] = var_s[k-1]*(1+Lambda[k-1]*dt)**2 + seqC[k]**2 * dt**2 * self.temporalDiscretization * self.sigma2s 
        
        # Overall variance per timestep
        # sigma2dt = var_a + var_i + var_s; # gaussian adding variance  
        
        return var_a + var_i + var_s
    
    def _compute_variance_fast(self, seqC, dt, Lambda):
        return cython_compute.compute_variance_parallel(
            seqC, dt, Lambda, self.sigma2a, self.sigma2i, self.sigma2s, self.temporalDiscretization
        )
    
    def _sensory_adaptation_for_seqC(self, seqC):
        """ sensory adaptation for the input sequences seqC 
            with the Ssame, Soppo, Selse parameters
        """
        lenC = len(seqC)
        # sensory adaptation for Snew
        if np.isnan(self.Selse).all(): # check if Selse is full of nan
            SelseTemp = np.ones((lenC))
        else:
            tempBins = np.arange(0, self.dt*(self.maxClength-1), self.dt)  # in matlab, the code was tempDiscrBins = 0:dt:dt*(maxClength-2) including the maxClength-2 element 0-13dt
                                        # because first pulse is always 0, nothing to sensory adapt there,
                                        # not by multiplication at least (which is what we do).                                        
            Selse = self.Selse[~np.isnan(self.Selse)] # remove nan
            order = np.prod(Selse.shape) - 1 # order of the polynomial
            
            orthogonalPolynomials = self._get_orthogonal_polynomials_curves(tempBins, order)
            SelseTemp = Selse @ orthogonalPolynomials[:order+1, :] # e.g. (1, 8) @ (8, 14) = (1, 14)
            # SelseTemp = SelseTemp.flatten()
            
        # sensory adaptation for Ssame and Soppo
        tempBins = np.arange(0, self.dt*(self.maxClength-2), self.dt) # in matlab, the code was tempDiscrBins = 0:dt:dt*(maxClength-3) including the maxClength-3 element 0-12dt
                                        # because our sequences always start from 0 =>
                                        # second pulse is either 0 or 'else' ([0 1... or [0 -1...) so 'same' and
                                        # 'opposite' can only start working from pulse 3 in the sequence.
        
        Ssame = self.Ssame[~np.isnan(self.Ssame)] # remove nan
        Soppo = self.Soppo[~np.isnan(self.Soppo)] # remove nan
        order = np.max([len(Ssame), len(Soppo)]) - 1 # order of the polynomial
        
        orthogonalPolynomials = self._get_orthogonal_polynomials_curves(tempBins, order)
        
        # order = len(Ssame) - 1
        # SsameTemp = Ssame @ orthogonalPolynomials[:order+1, :]
        # order = len(Soppo) - 1
        # SoppoTemp = Soppo @ orthogonalPolynomials[:order+1, :]
        
        SsameTemp = Ssame @ orthogonalPolynomials[:len(Ssame), :]
        SoppoTemp = Soppo @ orthogonalPolynomials[:len(Soppo), :]
        
        # SsameTemp = SsameTemp.flatten()
        # SoppoTemp = SoppoTemp.flatten()
        
        # operate sensory adaptation
        adaptedC = np.zeros(seqC.shape)
        countSame, countOppo, countElse = 0, 0, 0
        
        for ii in np.arange(1, lenC):
        
            if (seqC[ii-1] == seqC[ii]) and (np.abs(seqC[ii]) > 0): # ++ or --, not 0
                countSame += 1
                countOppo, countElse= 0, 0
                adaptedC[ii] = SsameTemp[countSame-1] * seqC[ii]
        
            elif (seqC[ii-1] == -seqC[ii]) and (np.abs(seqC[ii]) > 0): # +- or -+, not 0
                countOppo += 1
                countSame, countElse = 0, 0
                adaptedC[ii] = SoppoTemp[countOppo-1] * seqC[ii]
            
            else:
                if seqC[ii-1] == 0:
                    countElse += 1
                    adaptedC[ii] = SelseTemp[countElse-1] * seqC[ii]
                else:
                    countElse = 0 
                    adaptedC[ii] = seqC[ii]
                countSame, countOppo = 0, 0
                
        return adaptedC
    
    # def _sensory_adaptation_for_seqC_fast(self, seqC):
    #     return cython_compute._sensory_adaptation_for_seqC_cy(
    #         seqC, self.Ssame, self.Soppo, self.Selse, self.dt, self.maxClength
    #     )
            
    def _compute_mean_trace(self, seqC, B, Lambda, dt):
        ''' compute the mean trace of the process (with no sigma involved)
        
        args:
            seqC:   input sequence
            B:      boundary
            Lambda: decay/leakage
            dt:     time step
        
        returns:
            a:      the mean trace
        '''
        a  = np.zeros((len(seqC)+1))
        da = np.zeros((len(seqC)))
        a[0] = 0
        
        for k in range(1, len(seqC)+1):
            if np.abs(a[k-1]) >= B[k-1]: # hit sticky boundary 
                a[k-1:] = B[k-1]*np.sign(a[k-1])
                break
            else:
                da[k-1] = seqC[k-1]*dt + Lambda[k-1]*a[k-1]*dt
                a[k]    = a[k-1] + da[k-1]
        
        if np.abs(a[k]) >= B[k-1]: # sticky boundary at the end
            a[k] = B[k-1] * np.sign(a[k])
        
        return a
    
    def _compute_mean_trace_fast(self, seqC, B, Lambda, dt):
        return cython_compute.compute_mean_trace_parallel(
            seqC, B, Lambda, dt
        )
    
    def _compute_stoch_trace(self, seqC, B, Lambda, dt, debug=False):
        ''' compute the stochastic trace of the process (with sigma involved)
        '''
        
        # dW  = np.sqrt(dt) * np.random.randn(len(seqC)) 
        rn = norm.ppf(np.random.rand(len(seqC))) if debug else np.random.randn(len(seqC))
        dW  = np.sqrt(dt) * rn
        
        rn = norm.ppf(np.random.rand(len(seqC))) if debug else np.random.randn(len(seqC))
        # eta = 1 + np.sqrt(self.sigma2s) * np.random.randn(len(seqC))          * np.sqrt(self.temporalDiscretization)
        eta = 1 + np.sqrt(self.sigma2s) * rn * np.sqrt(self.temporalDiscretization)
        
        
        rn = norm.ppf(np.random.rand(1)) if debug else np.random.randn(1)
        a    = np.zeros((len(seqC)+1))
        da   = np.zeros((len(seqC)))
        # a[0] = np.sqrt(self.sigma2i) * np.random.randn(1) 
        a[0] = np.sqrt(self.sigma2i) * rn
        
        for k in range(1, len(seqC)+1):
            if np.abs(a[k-1]) >= B[k-1]: # hit sticky boundary 
                a[k-1:] = B[k-1]*np.sign(a[k-1])
                break
            else:
                da[k-1] = np.sqrt(self.sigma2a) * dW[k-1] + seqC[k-1]*eta[k-1]*dt + Lambda[k-1]*a[k-1]*dt
                a[k]    = a[k-1] + da[k-1]
            
        if np.abs(a[k]) >= B[k-1]: # sticky boundary at the end
            a[k] = B[k-1] * np.sign(a[k])
        
        return a
        
    def _compute_stoch_trace_fast(self, seqC, B, Lambda, dt, debug=False):
        # dW  = np.sqrt(dt) * np.random.randn(len(seqC)) 
        rn = norm.ppf(np.random.rand(len(seqC))) if debug else np.random.randn(len(seqC))
        dW  = np.sqrt(dt) * rn
        
        rn = norm.ppf(np.random.rand(len(seqC))) if debug else np.random.randn(len(seqC))
        # eta = 1 + np.sqrt(self.sigma2s) * np.random.randn(len(seqC))          * np.sqrt(self.temporalDiscretization)
        eta = 1 + np.sqrt(self.sigma2s) * rn * np.sqrt(self.temporalDiscretization)
        
        
        rn = norm.ppf(np.random.rand(1)) if debug else np.random.randn(1)
        a    = np.zeros((len(seqC)+1))
        da   = np.zeros((len(seqC)))
        # a[0] = np.sqrt(self.sigma2i) * np.random.randn(1) 
        a[0] = np.sqrt(self.sigma2i) * rn
        
        return cython_compute.compute_stoch_trace_parallel(
            dW, eta, a, da, seqC, B, Lambda, dt, self.sigma2a, self.sigma2i, self.sigma2s, self.temporalDiscretization, debug
        )
        
    def _affine_transformation(self, arrayIn):
        """affine transformation of the input array into range [-1,1]
        """
        return 2*(arrayIn - np.min(arrayIn))/(np.max(arrayIn) - np.min(arrayIn)) - 1
    
    def _affine_transformation_fast(self, arrayIn):
        return cython_compute._affine_transformation_cy(arrayIn)
    
    def _get_orthogonal_polynomials_curves(self, arrayIn, order):
        """get orthogonal polynomials matrix of shape (order, len(arrayIn))
            here highest order is set as 6
            order 0: 1
            order 1: 1*arrayIn
            order 2: 3*arrayIn**2 - 1
            order 3: 1/2*(5*arrayIn**3 - 3*arrayIn)
            order 4: 1/8*(35*arrayIn**4 - 30*arrayIn**2 + 3)
            order 5: 1/8*(63*arrayIn**5 - 70*arrayIn**3 + 15*arrayIn)
            order 6: 1/16*(231*arrayIn**6 - 315*arrayIn**4 + 105*arrayIn**2 - 5)
            order 7: 1/16*(429*arrayIn**7 - 693*arrayIn**5 + 315*arrayIn**3 - 35*arrayIn)

            args:
                arrayIn: input array
                order: order of the polynomial
            return:
                matrixOut: orthogonal polynomials matrix of shape (8, len(arrayIn))
        """
        # the input order should be less than 7
        if order > 7:
            raise Exception("order should be less than 7")
        
        matrixOut = np.zeros((8, len(arrayIn)))
        
        arrayInAffine = self._affine_transformation(arrayIn)
        
        if order >= 0:
            matrixOut[0,:] = np.ones(len(arrayIn))
        if order >= 1:
            matrixOut[1,:] = arrayInAffine
        if order >= 2:
            matrixOut[2,:] = 1/2*(3*arrayInAffine**2 - 1)
        if order >= 3:
            matrixOut[3,:] = 1/2*(5*arrayInAffine**3 - 3*arrayInAffine)
        if order >= 4:
            matrixOut[4,:] = 1/8*(35*arrayInAffine**4 - 30*arrayInAffine**2 + 3)
        if order >= 5:
            matrixOut[5,:] = 1/8*(63*arrayInAffine**5 - 70*arrayInAffine**3 + 15*arrayInAffine)
        if order >= 6:
            matrixOut[6,:] = 1/16*(231*arrayInAffine**6 - 315*arrayInAffine**4 + 105*arrayInAffine**2 - 5)
        if order >= 7:
            matrixOut[7,:] = 1/16*(429*arrayInAffine**7 - 693*arrayInAffine**5 + 315*arrayInAffine**3 - 35*arrayInAffine)
        
        return matrixOut
    
    def _get_orthogonal_polynomials_curves_fast(self, arrayIn, order):
        return cython_compute._get_orthogonal_polynomials_curves_cy(arrayIn, order)

if __name__ == "__main__":
    params = [0.00993, 0.00430, 0.00000, 1.33501, 3.57432]
    model_name = 'B-G-L0S-O-N-'
    
    # input parameters in the format of array
    # from simulator.DM_model import DM_model
    model = DM_model(params, model_name=model_name)
    print('create model from array and model name')
    print(f"Params: {params}")
    print(f"Bias: {model.bias}")
    print(f"Sigmas: {model.sigmas}")
    print(f"BGLSON: \n{model.BGLSON}")
    print(f"Model name: {model.model_name}")
    print()
    
    seqC = np.array([0, 0.4, -0.4, np.nan, np.nan,\
                    np.nan, np.nan, np.nan, np.nan, np.nan,\
                    np.nan, np.nan, np.nan, np.nan, np.nan])
    _, probR = model.stoch_simulation(seqC)

    # run simulation given input seqC
    seqC = [0, -0.2, -0.2, 0.2, -0.2,  0.2,  0.2,    0, -0.2, 0.2, -0.2, 0,   -0.2, 0, -0.2]
    a, probR = model.simulate(seqC)
    print(a, probR)
    
    import sys
    sys.path.append('./src')
    # input parameters in the format of dictionary
    from parse_data.decode_parameter import decode_mat_fitted_parameters
    paramsFitted = decode_mat_fitted_parameters('../data/params/263 models fitPars/data_fitPars_S1.mat')
    idx = 132
    params = dict(bias = paramsFitted['bias'][idx],
                sigmas = paramsFitted['sigmas'][idx,:],
                BGLS = paramsFitted['BGLS'][idx, :, :], 
                model_name = paramsFitted['allModelsList'][idx])
                
    model = DM_model(params)
    print('create model from dictionary')
    print(f"Bias: {model.bias}")
    print(f"Sigmas: {model.sigmas}")
    print(f"BGLSON: \n{model.BGLSON}")
    print(f"Model name: {model.model_name}")
    seqC = [0, -0.2, -0.2, 0.2, -0.2,  0.2,  0.2,    0, -0.2, 0.2, -0.2, 0,   -0.2, 0, -0.2]
    a, probR = model.simulate(seqC)
    print(a, probR)