# %%cython
# ext_kwargs={'extra_link_args': ['-Wno-error=unused-command-line-argument']}

# Add this at the top of your Cython code
cdef extern from *:
    """
    #define abort() ((void) 0)
    """


# cython stoch/mean trace computation
# compute variance
cimport cython
# cimport openmp
from cython.parallel import prange
from libc.math cimport copysign, fabs, sqrt
import numpy as np
from scipy.stats import norm

# cdef extern from "omp.h":
#     int omp_get_thread_num()

def compute_mean_trace_parallel(seqC, B, Lambda, dt):
    # check data type
    assert seqC.dtype   == np.float64
    assert B.dtype      == np.float64
    assert Lambda.dtype == np.float64
    # assert dt.dtype     == np.float64

    # check ndim
    assert seqC.ndim    == 1
    assert B.ndim       == 1
    assert Lambda.ndim  == 1

    # check shape

    # check whether array is contiguous
    assert seqC.flags['C_CONTIGUOUS']
    assert B.flags['C_CONTIGUOUS']
    assert Lambda.flags['C_CONTIGUOUS']

    # allocate output and state variables
    N = seqC.shape[0]
    a = np.zeros((N+1), dtype=np.float64)
    da = np.zeros((N), dtype=np.float64)
    a[0] = 0.0
    
    _compute_mean_trace_parallel(seqC, B, Lambda, dt, a, da)
    return a

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void update_da_a_mean(
    double[::1]   seqC,
    double[::1]   Lambda,
    double        dt,
    double[::1]   a,
    double[::1]   da,
    Py_ssize_t    k
) nogil:

    cdef double C_dt = seqC[k]*dt
    cdef double Lambda_a_dt = Lambda[k]*a[k]*dt
    
    da[k]  = C_dt + Lambda_a_dt
    a[k+1] = a[k] + da[k]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _compute_mean_trace_parallel(
    double[::1]   seqC,
    double[::1]   B,
    double[::1]   Lambda,
    double        dt,
    double[::1]   a,
    double[::1]   da
):

    cdef Py_ssize_t N = seqC.shape[0]
    cdef Py_ssize_t k

    cdef double abs_a, sign_a

    # with nogil, openmp.omp_set_num_threads(16):
    # openmp.omp_set_num_threads(16)
    for k in prange(1, N+1, nogil=True):
        abs_a = fabs(a[k-1])
        sign_a = copysign(1.0, a[k-1])

        if abs_a >= B[k-1]: # hit sticky boundary 
            a[k-1:] = B[k-1] * sign_a # set the rest of the trace to the sticky boundary value
            break
        else:
            # with boundscheck(True), wraparound(True):
            update_da_a_mean(seqC, Lambda, dt, a, da, k-1)
    
    abs_a = fabs(a[k])
    sign_a = copysign(1.0, a[k])

    if abs_a >= B[k-1]: # sticky boundary at the end
        a[k] = B[k-1] * sign_a


def compute_stoch_trace_parallel( dW, eta, a, da, seqC, B, Lambda, dt, sigma2a, sigma2i, sigma2s, temporalDiscretization, debug=False):
    """ 
    """
    # check dtype
    assert seqC.dtype   == np.float64
    assert B.dtype      == np.float64
    assert Lambda.dtype == np.float64
    assert dW.dtype     == np.float64
    assert eta.dtype    == np.float64
    assert a.dtype      == np.float64
    assert da.dtype     == np.float64
    
    # check ndim
    assert seqC.ndim    == 1
    assert B.ndim       == 1
    assert Lambda.ndim  == 1
    assert dW.ndim      == 1
    assert eta.ndim     == 1
    assert a.ndim       == 1
    assert da.ndim      == 1
    
    # check shape
    
    # check whether array is contiguous
    assert seqC.flags['C_CONTIGUOUS']
    assert B.flags['C_CONTIGUOUS']
    assert Lambda.flags['C_CONTIGUOUS']
    assert dW.flags['C_CONTIGUOUS']
    assert eta.flags['C_CONTIGUOUS']
    assert a.flags['C_CONTIGUOUS']
    assert da.flags['C_CONTIGUOUS']    
    
    # allocate output and state variables
    # cdef np.ndarray[np.float64, ndim=1] rn1, rn2, rn3
    # cdef Py_ssize_t N = seqC.shape[0]
    N = seqC.shape[0]

    # computation in C
    _compute_stoch_trace_parallel(seqC, B, Lambda, dt, sigma2a, sigma2i, sigma2s, temporalDiscretization, a, da, dW, eta)
    
    return a

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void update_da_a_stoch(
    double[::1]   seqC,
    double[::1]   Lambda,
    double        dt,
    double[::1]   a,
    double[::1]   da,
    Py_ssize_t    k,
    double[::1]   dW,
    double[::1]   eta,
    double        sigma2a,
) nogil:

    cdef double part1 = sqrt(sigma2a) * dW[k]
    cdef double part2 = seqC[k] * eta[k] * dt
    cdef double part3 = Lambda[k] * a[k] * dt

    da[k]  = part1 + part2 + part3
    a[k+1] = a[k] + da[k]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _compute_stoch_trace_parallel(
    double[::1]   seqC,
    double[::1]   B,
    double[::1]   Lambda,
    double        dt,
    double        sigma2a,
    double        sigma2i,
    double        sigma2s,
    int           temporalDiscretization,
    double[::1]   a,
    double[::1]   da,
    double[::1]   dW,
    double[::1]   eta
):
    cdef Py_ssize_t N = seqC.shape[0]
    cdef Py_ssize_t k

    cdef double abs_a, sign_a

    for k in prange(1, N+1, nogil=True):
        abs_a = fabs(a[k-1])
        sign_a = copysign(1.0, a[k-1])

        if abs_a >= B[k-1]: # hit sticky boundary 
            a[k-1:] = B[k-1] * sign_a # set the rest of the trace to the sticky boundary value
            break
        else:
            # with boundscheck(True), wraparound(True):
            update_da_a_stoch(seqC, Lambda, dt, a, da, k-1, dW, eta, sigma2a)
    
    abs_a = fabs(a[k])
    sign_a = copysign(1.0, a[k])

    if abs_a >= B[k-1]: # sticky boundary at the end
        a[k] = B[k-1] * sign_a

def compute_variance_parallel(seqC, dt, Lambda, sigma2a, sigma2i, sigma2s, temporalDiscretization):
    """ 
    """
    # check data type
    assert seqC.dtype == np.float64
    assert Lambda.dtype == np.float64

    # check ndim
    assert seqC.ndim == 1
    assert Lambda.ndim == 1
    
    # check shape
    assert seqC.shape == Lambda.shape

    # check whether array is contiguous
    assert seqC.flags['C_CONTIGUOUS']
    assert Lambda.flags['C_CONTIGUOUS']
    
    # allocate output and state variables
    N = seqC.shape[0]
    var_a  = np.zeros((N), dtype=np.float64)
    var_a[0] =  sigma2a * dt

    var_i = np.zeros((N+1), dtype=np.float64)
    var_i[0] = sigma2i

    var_s = np.zeros((N), dtype=np.float64)
    var_s[0] = seqC[0]**2 * sigma2s * dt

    _compute_variance_parallel(seqC, dt, Lambda, sigma2a, sigma2i, sigma2s, temporalDiscretization, var_a, var_i, var_s)

    var_i2 = var_i[1:]
    
    return var_a + var_i2 + var_s


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void update_var_a(
    double[::1]   Lambda,
    double        sigma2a,
    double        dt,
    double[::1]   var_a,
    Py_ssize_t    k,
)nogil:  
    cdef double part1 = var_a[k-1]
    cdef double part2 = Lambda[k-1] * dt
    cdef double part3 = sigma2a * dt
    cdef double part4 = part1*(1+part2)**2

    var_a[k] = part4 + part3

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void update_var_i(
    double[::1]   Lambda,
    double        dt,
    double[::1]   var_i,
    Py_ssize_t    k,
)nogil:  
    cdef double part1 = var_i[k-1]
    cdef double part2 = Lambda[k-1] * dt + 1
    cdef double part3 = part2**2
    
    var_i[k] = part1*part3

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void update_var_s(
    double[::1]   seqC,
    double        dt,
    double[::1]   Lambda,
    double        sigma2s,
    int           temporalDiscretization,
    double[::1]   var_s,
    Py_ssize_t    k,
)nogil:
    cdef double part1 = var_s[k-1]
    cdef double part2 = seqC[k]**2 * dt**2 * temporalDiscretization * sigma2s 
    cdef double part3 = (1+Lambda[k-1]*dt)**2

    var_s[k] = part1*part3 + part2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _compute_variance_parallel(
    double[::1]   seqC,
    double        dt,
    double[::1]   Lambda,
    double        sigma2a,
    double        sigma2i,
    double        sigma2s,
    int           temporalDiscretization,
    double[::1]   var_a,
    double[::1]   var_i,
    double[::1]   var_s,
):
    cdef Py_ssize_t N = seqC.shape[0]
    cdef Py_ssize_t k
    
    for k in prange(1, N, nogil=True):
        update_var_a(Lambda, sigma2a, dt, var_a, k)
        
    for k in prange(1, N+1, nogil=True):
        update_var_i(Lambda, dt, var_i, k)
    # var_i = var_i[1:]

    for k in prange(1, N, nogil=True):
        update_var_s(seqC, dt, Lambda,  sigma2s, temporalDiscretization, var_s, k)
    