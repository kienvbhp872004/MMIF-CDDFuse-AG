import numpy as np

def calculate_ncc(im1, im2):
    """
    Nonlinear Correlation Coefficient (NCC) matching NCC_inline in ncie.m
    """
    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)
    
    # Joint histogram
    # Using np.histogram2d for speed
    h, _, _ = np.histogram2d(im1.flatten(), im2.flatten(), bins=256, range=[[0, 256], [0, 256]])
    
    # Normalize to probability
    h = h / (np.sum(h) + 1e-12)
    
    # Marginal distributions
    p1 = np.sum(h, axis=1)
    p2 = np.sum(h, axis=0)
    
    # Entropy (normalized by log2(256) = 8)
    def entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log2(p)) / 8.0
    
    h_x = entropy(p1)
    h_y = entropy(p2)
    
    # Joint entropy
    h_xy = h.flatten()
    h_xy = h_xy[h_xy > 0]
    h_joint = -np.sum(h_xy * np.log2(h_xy)) / 8.0
    
    # NCC = H_x + H_y - H_xy
    return h_x + h_y - h_joint

def ncie(im1, im2, fim):
    """
    Normalized Correlation Information Entropy (NCIE) matching ncie.m
    """
    def normalize_inline(data):
        data = data.astype(np.float64)
        da = np.max(data)
        xiao = np.min(data)
        if da == xiao:
            return np.round(data * 0) # Constant image
        else:
            return np.round(((data - xiao) / (da - xiao)) * 255)

    im1_n = normalize_inline(im1)
    im2_n = normalize_inline(im2)
    fim_n = normalize_inline(fim)
    
    # NCC computations
    ncc_xy = calculate_ncc(im1_n, im2_n)
    ncc_xf = calculate_ncc(im1_n, fim_n)
    ncc_yf = calculate_ncc(im2_n, fim_n)
    
    # Correlation matrix
    R = np.array([
        [1.0, ncc_xy, ncc_xf],
        [ncc_xy, 1.0, ncc_yf],
        [ncc_xf, ncc_yf, 1.0]
    ])
    
    # Eigenvalues
    r = np.linalg.eigvals(R)
    # Eigenvalues can be slightly complex due to precision, but R is symmetric so they should be real.
    r = np.real(r)
    
    # NCIE calculation
    K = 3.0
    b = 256.0
    
    # HR = sum(r .* log2(r ./ K) / K)
    # Avoid log of zero or negative (though eigenvalues of R should be >= 0)
    r = np.clip(r, 1e-15, None)
    hr = np.sum(r * np.log2(r / K) / K)
    
    # Normalize by log2(b) = 8
    hr_norm = -hr / 8.0
    
    return 1 - hr_norm
