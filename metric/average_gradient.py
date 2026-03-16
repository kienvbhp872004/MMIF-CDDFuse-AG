import numpy as np

def average_gradient(F):
    """
    Average Gradient (Standard version matching Matlab metricsAvg_gradient.m)
    """
    F = F.astype(np.float64)
    # np.gradient uses central differences for interior points and 
    # first-order differences at boundaries, matching Matlab's gradient().
    # Note: np.gradient returns (dy, dx) for 2D arrays.
    dy, dx = np.gradient(F)
    
    s = np.sqrt((dx**2 + dy**2) / 2.0)
    r, c = F.shape
    # The Matlab code uses (r-1)*(c-1) as denominator
    res = np.sum(s) / ((r - 1) * (c - 1))
    return res
