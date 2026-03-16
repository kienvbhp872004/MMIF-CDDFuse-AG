import numpy as np

def custom_mse(a, b):
    # This matches the 'mse' subfunction in Matlab's metricsRmse.m
    # res0 = sqrt(sum((a-b)^2)) / (m*n)
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    m, n = a.shape
    
    temp = np.sqrt(np.sum((a - b)**2))
    return temp / (m * n)

def rmse(A, B, F):
    """
    RMSE (Standard version matching Matlab metricsRmse.m)
    Note: The Matlab version uses a custom 'mse' definition.
    """
    rmse_af = custom_mse(A, F)
    rmse_bf = custom_mse(B, F)
    
    return (rmse_af + rmse_bf) / 2.0
