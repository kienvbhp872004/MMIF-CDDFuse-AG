import numpy as np

def variance(F):
    """
    Standard Deviation (Matching Matlab metricsVariance.m)
    Note: Despite the name, metricsVariance.m in the MATLAB folder returns SD.
    """
    return np.std(F.astype(np.float64))
