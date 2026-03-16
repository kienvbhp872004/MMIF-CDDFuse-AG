import numpy as np

def custom_mse(a, b):
    # Matches the 'mse' subfunction in Matlab's metricsPsnr.m
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    m, n = a.shape
    
    # temp = sqrt(sum((a-b)^2))
    temp = np.sqrt(np.sum((a - b)**2))
    # res0 = temp / (m*n)
    return temp / (m * n)

def psnr(A, B, F):
    """
    PSNR (Standard version matching Matlab metricsPsnr.m)
    Note: Highly non-standard MSE and PSNR definition in this specific folder.
    """
    MAX = 255.0
    
    # MES = (mse(img1, fused) + mse(img2, fused)) / 2.0
    # Note: the Matlab code calls it 'MES' (likely typo for MSE)
    mes = (custom_mse(A, F) + custom_mse(B, F)) / 2.0
    
    # PSNR = 20 * log10(MAX / sqrt(MES))
    if mes == 0:
        return float('inf')
        
    psnr = 20 * np.log10(MAX / np.sqrt(mes))
    return psnr
