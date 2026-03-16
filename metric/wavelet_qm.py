import numpy as np

def haar_2d_one_level(img):
    """
    Simulate one level of 2D Haar wavelet decomposition using numpy.
    Returns (LL, (LH, HL, HH))
    """
    h, w = img.shape
    # Ensure even dimensions
    img = img[:h - (h % 2), :w - (w % 2)]
    
    # Reshape and sum/diff
    # LL: (A+B+C+D)/2
    # HL: (A-B+C-D)/2
    # LH: (A+B-C-D)/2
    # HH: (A-B-C+D)/2
    # Actually Haar coefficients are often scaled by 1/sqrt(2) per dimension.
    
    # Simple pooling approach
    A = img[0::2, 0::2]
    B = img[0::2, 1::2]
    C = img[1::2, 0::2]
    D = img[1::2, 1::2]
    
    LL = (A + B + C + D) / 2.0
    HL = (A - B + C - D) / 2.0
    LH = (A + B - C - D) / 2.0
    HH = (A - B - C + D) / 2.0
    
    return LL, (LH, HL, HH)

def wavelet_decompose(img, levels=2):
    coeffs = []
    current_ll = img
    for i in range(levels):
        current_ll, det = haar_2d_one_level(current_ll)
        coeffs.append(det)
    coeffs.reverse() # To match pywt order: [LL_n, (LH_n, HL_n, HH_n), ..., (LH_1, HL_1, HH_1)]
    # Actually pywt returns [LL, (LHn, HLn, HHn), ..., (LH1, HL1, HH1)]
    # But wait, level 1 is highest freq.
    # Level n is lowest freq.
    # My logic: 
    # coeffs[0] is level 2 det (if levels=2)
    # coeffs[1] is level 1 det
    # We need LL at the very beginning.
    return [current_ll] + coeffs

def high_frequency_energy(LH, HL, HH):
    return LH**2 + HL**2 + HH**2

def edge_preservation(A_band, F_band):
    return np.exp(-np.abs(A_band - F_band))

def QM(A, B, F, levels=2, beta=None):
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    F = F.astype(np.float32)
    if beta is None:
        beta = [1] * levels
    coeffA = wavelet_decompose(A, levels)
    coeffB = wavelet_decompose(B, levels)
    coeffF = wavelet_decompose(F, levels)
    Q_scales = []
    # pywt order: coeff[0] is LL, coeff[1] is scale n (lowest freq det), ..., coeff[n] is scale 1 (highest freq det)
    # We iterate from scale 1 to n.
    for s in range(1, levels + 1):
        LH_A, HL_A, HH_A = coeffA[s]
        LH_B, HL_B, HH_B = coeffB[s]
        LH_F, HL_F, HH_F = coeffF[s]
        EP_AF = (edge_preservation(LH_A, LH_F) + edge_preservation(HL_A, HL_F) + edge_preservation(HH_A, HH_F)) / 3.0
        EP_BF = (edge_preservation(LH_B, LH_F) + edge_preservation(HL_B, HL_F) + edge_preservation(HH_B, HH_F)) / 3.0
        wA = high_frequency_energy(LH_A, HL_A, HH_A)
        wB = high_frequency_energy(LH_B, HL_B, HH_B)
        numerator = np.sum(EP_AF * wA + EP_BF * wB)
        denominator = np.sum(wA + wB) + 1e-12
        Qs = numerator / denominator
        Q_scales.append(Qs)
    QM_res = 1.0
    for s in range(levels):
        QM_res *= Q_scales[s] ** beta[s]
    return QM_res