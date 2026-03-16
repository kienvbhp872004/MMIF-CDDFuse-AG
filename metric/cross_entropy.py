import numpy as np

def calculate_cross_entropy(img, fused):
    # Ensure images are in [0, 255] range and uint8-like for tiling
    img = np.round(img).astype(np.int32)
    fused = np.round(fused).astype(np.int32)
    
    m, n = img.shape
    
    # Calculate histograms
    h1 = np.histogram(img, bins=256, range=(0, 256))[0]
    h2 = np.histogram(fused, bins=256, range=(0, 256))[0]
    
    p1 = h1 / (m * n)
    p2 = h2 / (m * n)
    
    # Cross entropy calculation: sum(P1 * log2(P1/P2))
    # Only for indices where both probabilities are non-zero
    mask = (p1 > 0) & (p2 > 0)
    res = np.sum(p1[mask] * np.log2(p1[mask] / p2[mask]))
    
    return res

def cross_entropy(A, B, F):
    """
    Cross Entropy (Standard version matching Matlab metricsCross_entropy.m)
    """
    ce_af = calculate_cross_entropy(A, F)
    ce_bf = calculate_cross_entropy(B, F)
    return (ce_af + ce_bf) / 2.0
