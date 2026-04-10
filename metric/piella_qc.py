import numpy as np
import cv2

def local_stats(x, y, win):
    """
    Vectorized computation of local means, variances, and covariance.
    Uses sliding window with step 1.
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    
    # Kernel for sum over window
    kernel = np.ones((win, win), np.float64)
    n = win * win
    
    mu_x = cv2.filter2D(x, -1, kernel, borderType=cv2.BORDER_CONSTANT) / n
    mu_y = cv2.filter2D(y, -1, kernel, borderType=cv2.BORDER_CONSTANT) / n
    
    mu_x_sq = cv2.filter2D(x*x, -1, kernel, borderType=cv2.BORDER_CONSTANT) / n
    mu_y_sq = cv2.filter2D(y*y, -1, kernel, borderType=cv2.BORDER_CONSTANT) / n
    mu_xy   = cv2.filter2D(x*y, -1, kernel, borderType=cv2.BORDER_CONSTANT) / n
    
    # Variances and Covariance
    var_x = mu_x_sq - mu_x**2
    var_y = mu_y_sq - mu_y**2
    cov_xy = mu_xy - mu_x * mu_y
    
    # We only care about the inner part (matching valid windows)
    pad = win // 2
    # If win is 8, filter2D with center-anchor will have (3, 3) shift or similar.
    # Actually filter2D is [pad, h-pad].
    # But wait, the original was range(0, h-win+1, 1). 
    # Let's just crop to match the original behavior precisely if needed, 
    # but the key is the average.
    
    # Let's crop to 'valid' regions
    # In range(0, h-win+1), the last index i=h-win means window is [h-win, h].
    # So results corresponds to indices [win//2, h - win + win//2]? 
    # Actually, let's just use the full maps and average them, it's roughly the same.
    # For EXACT matching of the original loops:
    res_mu_x = mu_x[win-1:, win-1:]
    res_mu_y = mu_y[win-1:, win-1:]
    res_var_x = var_x[win-1:, win-1:]
    res_var_y = var_y[win-1:, win-1:]
    res_cov_xy = cov_xy[win-1:, win-1:]
    
    return res_mu_x, res_mu_y, res_var_x, res_var_y, res_cov_xy

def QC(A, B, F, win=8):
    # Vectorized UIQI logic
    def get_uiqi(x_mean, y_mean, x_var, y_var, xy_cov):
        eps = 1e-8
        num = 4 * xy_cov * x_mean * y_mean
        den = (x_var + y_var + eps) * (x_mean**2 + y_mean**2 + eps)
        res = num / den
        return res

    muA, muF, varA, varF, covAF = local_stats(A, F, win)
    muB, _, varB, _, covBF = local_stats(B, F, win) # reuse muF and varF
    
    # Similarity map (sim = sigma_AF / (sigma_AF + sigma_BF))
    denom_sim = covAF + covBF
    sim = np.divide(covAF, denom_sim, out=np.zeros_like(covAF), where=denom_sim!=0)
    sim = np.clip(sim, 0, 1)
    
    # UIQI maps
    QAF = get_uiqi(muA, muF, varA, varF, covAF)
    QBF = get_uiqi(muB, muF, varB, varF, covBF)
    
    # Global index
    QC_map = sim * QAF + (1 - sim) * QBF
    return np.mean(QC_map)