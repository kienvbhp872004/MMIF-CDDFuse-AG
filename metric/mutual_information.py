import numpy as np

def calculate_mutinf(a, b):
    # Normalized image to [0, 255] as per Matlab metricsMutinf.m
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    
    # Normalization logic from Matlab:
    if np.max(a) != np.min(a):
        a = (a - np.min(a)) / (np.max(a) - np.min(a))
    else:
        a = np.zeros_like(a)
        
    if np.max(b) != np.min(b):
        b = (b - np.min(b)) / (np.max(b) - np.min(b))
    else:
        b = np.zeros_like(b)
        
    a = (a * 255).astype(np.int16).astype(np.float64)
    b = (b * 255).astype(np.int16).astype(np.float64)
    
    # Use np.histogram2d to get the joint histogram
    # Range is [0, 255], bins=256
    hab, _, _ = np.histogram2d(a.flatten(), b.flatten(), bins=256, range=[[0, 256], [0, 256]])
    
    # Marginal histograms
    ha = np.sum(hab, axis=1)
    hb = np.sum(hab, axis=0)
    
    # Joint Entropy (Natural Log as per Matlab log())
    hsum = np.sum(hab)
    if hsum > 0:
        p_joint = hab / hsum
        hab_entropy = -np.sum(p_joint[p_joint > 0] * np.log(p_joint[p_joint > 0]))
    else:
        hab_entropy = 0
        
    # Marginal Entropies
    hsum_a = np.sum(ha)
    if hsum_a > 0:
        pa = ha / hsum_a
        ha_entropy = -np.sum(pa[pa > 0] * np.log(pa[pa > 0]))
    else:
        ha_entropy = 0
        
    hsum_b = np.sum(hb)
    if hsum_b > 0:
        pb = hb / hsum_b
        hb_entropy = -np.sum(pb[pb > 0] * np.log(pb[pb > 0]))
    else:
        hb_entropy = 0
        
    # Mutual Information
    mi = ha_entropy + hb_entropy - hab_entropy
    return mi

def mutual_information(A, B, F):
    """
    Mutual Information (Standard version matching Matlab metricsMutinf.m)
    Sum of MI(A, F) and MI(B, F)
    """
    mi_af = calculate_mutinf(A, F)
    mi_bf = calculate_mutinf(B, F)
    return mi_af + mi_bf
