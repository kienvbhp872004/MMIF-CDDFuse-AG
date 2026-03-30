import numpy as np
import cv2

def local_stats(x, y, win, step=8):
    """
    Vectorized computation of local means and variances for non-overlapping blocks.
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    
    # Kernel for sum over window
    kernel = np.ones((win, win), np.float64)
    n = win * win
    
    mu_x_full = cv2.filter2D(x, -1, kernel, borderType=cv2.BORDER_CONSTANT) / n
    mu_y_full = cv2.filter2D(y, -1, kernel, borderType=cv2.BORDER_CONSTANT) / n
    
    mu_x_sq_full = cv2.filter2D(x*x, -1, kernel, borderType=cv2.BORDER_CONSTANT) / n
    mu_y_sq_full = cv2.filter2D(y*y, -1, kernel, borderType=cv2.BORDER_CONSTANT) / n
    mu_xy_full   = cv2.filter2D(x*y, -1, kernel, borderType=cv2.BORDER_CONSTANT) / n
    
    # Extract blocks (every 'step')
    # Original loop for i in range(0, h-win+1, step)
    # The filter2D result map at [i + win-1, j + win-1] would give the sum of [i, i+win-1].
    mu_x = mu_x_full[win-1::step, win-1::step]
    mu_y = mu_y_full[win-1::step, win-1::step]
    mu_x_sq = mu_x_sq_full[win-1::step, win-1::step]
    mu_y_sq = mu_y_sq_full[win-1::step, win-1::step]
    mu_xy = mu_xy_full[win-1::step, win-1::step]
    
    # Variances and Covariance
    var_x = np.maximum(mu_x_sq - mu_x**2, 0)
    var_y = np.maximum(mu_y_sq - mu_y**2, 0)
    cov_xy = mu_xy - mu_x * mu_y
    
    return mu_x, mu_y, var_x, var_y, cov_xy

def uiqi_vec(x_mean, y_mean, x_var, y_var, xy_cov, eps=1e-10):
    numerator = 4 * xy_cov * x_mean * y_mean
    denominator = (x_var + y_var) * (x_mean**2 + y_mean**2) + eps
    return numerator / denominator

def QS(A, B, F, window_size=8):
    # Non-overlapping blocks (step = window_size)
    muA, muF, varA, varF, covAF = local_stats(A, F, window_size, step=window_size)
    muB, _, varB, _, covBF = local_stats(B, F, window_size, step=window_size)
    
    # Lambda = varA / (varA + varB)
    denom_lam = varA + varB
    lam = np.divide(varA, denom_lam, out=np.ones_like(varA)*0.5, where=denom_lam!=0)
    
    # UIQI maps
    qAF = uiqi_vec(muA, muF, varA, varF, covAF)
    qBF = uiqi_vec(muB, muF, varB, varF, covBF)
    
    # Global index
    qs_map = lam * qAF + (1 - lam) * qBF
    return np.mean(qs_map)
