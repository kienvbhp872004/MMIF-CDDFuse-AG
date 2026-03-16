import numpy as np

def entropy(F):
    """
    Entropy (Standard version matching Matlab metricsEntropy.m)
    """
    F = np.floor(F).astype(np.int32)
    m, n = F.shape
    
    # Histogram
    h = np.histogram(F, bins=256, range=(0, 256))[0]
    p = h / (m * n)
    
    # Filter non-zero probabilities
    p = p[p > 0]
    
    # Entropy calculation
    res = -np.sum(p * np.log2(p))
    return res
