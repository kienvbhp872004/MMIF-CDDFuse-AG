import numpy as np

def spatial_frequency(F):
    """
    Spatial Frequency (Standard version matching Matlab metricsSpatial_frequency.m)
    """
    F = F.astype(np.float64)
    m, n = F.shape
    
    # Row Frequency
    # RF = sqrt( sum((F(i,j) - F(i,j-1))^2) / (MN) )
    rf_sq_sum = np.sum((F[:, 1:] - F[:, :-1])**2)
    rf = rf_sq_sum / (m * n)
    
    # Column Frequency
    # CF = sqrt( sum((F(i,j) - F(i-1,j))^2) / (MN) )
    cf_sq_sum = np.sum((F[1:, :] - F[:-1, :])**2)
    cf = cf_sq_sum / (m * n)
    
    res = np.sqrt(rf + cf)
    return res
