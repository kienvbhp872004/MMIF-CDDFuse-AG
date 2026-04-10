import numpy as np
import cv2
from numpy.lib.stride_tricks import sliding_window_view

def fmi(ima, imb, imf, feature='none', w=3):
    """
    Feature Mutual Information (FMI) - Fully Vectorized implementation matching fmi.m
    Based on Mohammad Haghighat's FMI.
    """
    ima = ima.astype(np.float64)
    imb = imb.astype(np.float64)
    imf = imf.astype(np.float64)

    # Feature Extraction
    if feature == 'none':
        a_feat, b_feat, f_feat = ima, imb, imf
    elif feature == 'gradient':
        # Matching Matlab gradient(A)
        # For 2D, gradient(A) returns [GX, GY]
        # Using np.gradient which returns [GY, GX]
        gy_a, gx_a = np.gradient(ima)
        gy_b, gx_b = np.gradient(imb)
        gy_f, gx_f = np.gradient(imf)
        # Matlab gradient(A) returns the first output (GX in some versions, but 
        # based on VIFB usage, it's often the gradient magnitude or just the x-partial)
        # However, checking fmi.m's switch case, it's a single return value.
        # In Matlab, for 2D, f = gradient(A) returns central differences along the first dimension (rows).
        a_feat, b_feat, f_feat = gy_a, gy_b, gy_f
    else:
        a_feat, b_feat, f_feat = ima, imb, imf

    # Sliding window params
    radius = w // 2
    L = w * w
    
    # 1. Processing A and F
    fmi_af = _calculate_normalized_mi(a_feat, f_feat, w, L)
    # 2. Processing B and F
    fmi_bf = _calculate_normalized_mi(b_feat, f_feat, w, L)
    
    return np.mean((fmi_af + fmi_bf) / 2.0)

def _calculate_normalized_mi(feat1, feat2, w_size, L):
    # Sliding window view
    try:
        win1 = sliding_window_view(feat1, (w_size, w_size)).reshape(-1, L)
        win2 = sliding_window_view(feat2, (w_size, w_size)).reshape(-1, L)
    except:
        return np.array([0.0])

    # Local Normalization for each window
    def normalize_windows(win):
        w_min = win.min(axis=1, keepdims=True)
        w_max = win.max(axis=1, keepdims=True)
        # Note: Matlab code uses 1.0 if constant
        norm = np.where(w_max > w_min, (win - w_min) / (w_max - w_min + 1e-12), 1.0)
        return norm

    win1_n = normalize_windows(win1)
    win2_n = normalize_windows(win2)

    # Marginal PDFs (each window is a distribution)
    sum1 = win1_n.sum(axis=1, keepdims=True)
    pdf1 = np.where(sum1 > 0, win1_n / (sum1 + 1e-12), 0.0)
    sum2 = win2_n.sum(axis=1, keepdims=True)
    pdf2 = np.where(sum2 > 0, win2_n / (sum2 + 1e-12), 0.0)

    # Marginal CDFs
    cdf1 = np.cumsum(pdf1, axis=1)
    cdf2 = np.cumsum(pdf2, axis=1)

    # Pearson Correlation (between PDF vectors)
    m1 = pdf1.mean(axis=1, keepdims=True)
    m2 = pdf2.mean(axis=1, keepdims=True)
    pdf1_tmp = pdf1 - m1
    pdf2_tmp = pdf2 - m2
    num = (pdf1_tmp * pdf2_tmp).sum(axis=1)
    den = np.sqrt((pdf1_tmp**2).sum(axis=1) * (pdf2_tmp**2).sum(axis=1))
    c_coef = np.where(den > 0, num / den, 0.0)

    # Standard Deviations of PDF vectors (treated as samples 1..L)
    indices = np.arange(1, L + 1)
    e1 = (indices * pdf1).sum(axis=1)
    e2_1 = (indices**2 * pdf1).sum(axis=1)
    sd1 = np.sqrt(np.clip(e2_1 - e1**2, 0, None))
    
    e2 = (indices * pdf2).sum(axis=1)
    e2_2 = (indices**2 * pdf2).sum(axis=1)
    sd2 = np.sqrt(np.clip(e2_2 - e2**2, 0, None))

    # Upper correlation (Frechet bound)
    # COV_Up = sum_i sum_j (min(C1_i, C2_j) - C1_i * C2_j)
    # This is vectorizable: 
    # cdf1[:, :, None] and cdf2[:, None, :]
    # Total sum is sum_i(C1_i * (1-C1_i)) if cdf1==cdf2? No.
    
    # We use a memory efficient approach for large N
    # covUp = np.sum(np.minimum(cdf1[:, :, None], cdf1[:, None, :]) - cdf1[:, :, None] * cdf1[:, None, :], axis=(1,2))
    # Wait, the formula for covUp is for SAME distribution? No, it's bivariate.
    # covUp is for the case of maximal correlation between X and Y with given marginals.
    
    N = pdf1.shape[0]
    phi = np.zeros(N)
    
    # Process covUp in chunks to save memory
    chunk_size = 5000
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        c1_chunk = cdf1[i:end]
        c2_chunk = cdf2[i:end]
        
        # (chunk, L, L)
        m_ij = np.minimum(c1_chunk[:, :, None], c2_chunk[:, None, :])
        p_ij = c1_chunk[:, :, None] * c2_chunk[:, None, :]
        cov_chunk = np.sum(m_ij - p_ij, axis=(1, 2))
        
        sd_prod = sd1[i:end] * sd2[i:end]
        corr_up = np.where(sd_prod > 0, cov_chunk / (sd_prod + 1e-12), 0.0)
        phi[i:end] = np.where(corr_up > 0, c_coef[i:end] / (corr_up + 1e-12), 0.0)

    # Joint Probability Matrix and Joint Entropy
    # jpdf = phi * jpdfUp + (1-phi) * (pdf1 * pdf2)
    # jpdfUp is the derivative of M[i,j] = min(C1_i, C2_j)
    
    h_win = np.zeros(N)
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        phi_chunk = phi[i:end, None, None]
        pdf1_chunk = pdf1[i:end]
        pdf2_chunk = pdf2[i:end]
        cdf1_chunk = cdf1[i:end]
        cdf2_chunk = cdf2[i:end]
        
        M = np.minimum(cdf1_chunk[:, :, None], cdf2_chunk[:, None, :])
        jpdf_up = np.zeros_like(M)
        jpdf_up[:, 0, 0] = M[:, 0, 0]
        jpdf_up[:, 1:, 0] = M[:, 1:, 0] - M[:, :-1, 0]
        jpdf_up[:, 0, 1:] = M[:, 0, 1:] - M[:, 0, :-1]
        jpdf_up[:, 1:, 1:] = M[:, 1:, 1:] - M[:, :-1, 1:] - M[:, 1:, :-1] + M[:, :-1, :-1]
        
        jpdf = phi_chunk * jpdf_up + (1 - phi_chunk) * (pdf1_chunk[:, :, None] * pdf2_chunk[:, None, :])
        
        # Joint Entropy
        # Flatten L,L
        jpdf_f = jpdf.reshape(-1, L*L)
        # Use clip to avoid negative values from mixture model with negative correlation
        jpdf_f = np.clip(jpdf_f, 1e-15, 1.0)
        je = -np.sum(jpdf_f * np.log2(jpdf_f), axis=1)
        
        # Marginal Entropies
        h1 = -np.sum(pdf1_chunk * np.log2(pdf1_chunk + 1e-12), axis=1)
        h2 = -np.sum(pdf2_chunk * np.log2(pdf2_chunk + 1e-12), axis=1)
        
        mi = h1 + h2 - je
        # Normalized MI: 2*MI / (H1 + H2)
        h_win[i:end] = np.where(mi > 0, 2 * mi / (h1 + h2 + 1e-12), 0.0)

    # Special case: if a_win == f_win, FMI = 1
    # Check absolute equality
    is_eq = np.all(win1 == win2, axis=1)
    h_win[is_eq] = 1.0
    
    return h_win
