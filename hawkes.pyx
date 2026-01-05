# hawkes_core.pyx
import numpy as np
cimport numpy as np
from libc.math cimport exp, log

def hawkes_neg_log_likelihood_cython(double[:] params, double[:] t):
    """
    Cythonized implementation of Ogata's Recursive Algorithm.
    Speedup comes from C-level typing and removing Python overhead in the loop.
    """
    cdef double mu = params[0]
    cdef double alpha = params[1]
    cdef double beta = params[2]
    
    # Constraints check
    if mu <= 1e-5 or alpha <= 1e-5 or beta <= 1e-5:
        return 1e10
        
    cdef int n = t.shape[0]
    cdef double T_max = t[n-1]
    
    # 1. Integral Term
    # We use a memory view slice for numpy sum, but simple scalar math is faster
    # calculation: mu * T + (alpha/beta) * sum(1 - exp(-beta * (T - t)))
    cdef double integral = mu * T_max
    cdef double sum_exp = 0.0
    cdef int j
    
    for j in range(n):
        sum_exp += (1.0 - exp(-beta * (T_max - t[j])))
        
    integral += (alpha / beta) * sum_exp
    
    # 2. Sum of Logs (The Recursive Loop)
    cdef double sum_log_lambda = 0.0
    cdef double R = 0.0
    cdef double delta_t = 0.0
    cdef double current_lambda = 0.0
    cdef int i
    
    # First point
    sum_log_lambda += log(mu)
    
    # Main Loop - Pure C speed
    for i in range(1, n):
        delta_t = t[i] - t[i-1]
        
        # Recursive update
        R = exp(-beta * delta_t) * (R + alpha)
        
        current_lambda = mu + R
        sum_log_lambda += log(current_lambda)
        
    return -(sum_log_lambda - integral)