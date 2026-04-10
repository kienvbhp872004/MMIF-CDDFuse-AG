import numpy as np
from scipy.signal import convolve2d

def per_extn_im_fn(x, wsize):
    """
    Periodic extension of the given image in 4 directions.
    wsize should be odd.
    """
    hwsize = (wsize - 1) // 2
    p, q = x.shape
    xout_ext = np.zeros((p + wsize - 1, q + wsize - 1))
    xout_ext[hwsize: p + hwsize, hwsize: q + hwsize] = x

    # Row-wise periodic extension
    if wsize - 1 == hwsize + 1:
        xout_ext[0: hwsize, :] = xout_ext[2, :].reshape(1, -1)
        xout_ext[p + hwsize: p + wsize - 1, :] = xout_ext[-3, :].reshape(1, -1)

    # Column-wise periodic extension
    xout_ext[:, 0: hwsize] = xout_ext[:, 2].reshape(-1, 1)
    xout_ext[:, q + hwsize: q + wsize - 1] = xout_ext[:, -3].reshape(-1, 1)

    return xout_ext

def sobel_fn(x):
    """
    Sobel operators with periodic extension.
    """
    vtemp = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
    htemp = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8.0

    a = htemp.shape[0]
    x_ext = per_extn_im_fn(x, a)
    
    gv = convolve2d(x_ext, vtemp, mode='valid')
    gh = convolve2d(x_ext, htemp, mode='valid')

    return gv, gh

def nabf(I1, I2, f):
    """
    NABF (Artifact Preservation / Fusion Artifacts) 
    Implementation matching the user provided logic.
    """
    # Parameters for Petrovic Metrics Computation.
    Td = 2
    wt_min = 0.001
    Lg = 1.5
    Nrg = 0.9999
    kg = 19
    sigmag = 0.5
    Nra = 0.9995
    ka = 22
    sigmaa = 0.5

    xrcw = f.astype(np.float64)
    x1 = I1.astype(np.float64)
    x2 = I2.astype(np.float64)

    # Edge Strength & Orientation.
    gvA, ghA = sobel_fn(x1)
    gA = np.sqrt(ghA**2 + gvA**2)

    gvB, ghB = sobel_fn(x2)
    gB = np.sqrt(ghB**2 + gvB**2)

    gvF, ghF = sobel_fn(xrcw)
    gF = np.sqrt(ghF**2 + gvF**2)

    # Relative Edge Strength & Orientation.
    gAF = np.zeros(gA.shape)
    gBF = np.zeros(gB.shape)
    
    eps = 1e-12
    
    maskAF2 = (gA > gF)
    gAF = np.where(maskAF2, gF / (gA + eps), gA / (gF + eps))
    # Handle zero gradients
    maskAF_zero = (gA == 0) | (gF == 0)
    gAF[maskAF_zero] = 0.0
    
    maskBF2 = (gB > gF)
    gBF = np.where(maskBF2, gF / (gB + eps), gB / (gF + eps))
    maskBF_zero = (gB == 0) | (gF == 0)
    gBF[maskBF_zero] = 0.0

    aA = np.where((gvA == 0) & (ghA == 0), 0.0, np.arctan(gvA / (ghA + eps)))
    aB = np.where((gvB == 0) & (ghB == 0), 0.0, np.arctan(gvB / (ghB + eps)))
    aF = np.where((gvF == 0) & (ghF == 0), 0.0, np.arctan(gvF / (ghF + eps)))

    aAF = np.abs(np.abs(aA - aF) - np.pi / 2.0) * 2.0 / np.pi
    aBF = np.abs(np.abs(aB - aF) - np.pi / 2.0) * 2.0 / np.pi

    QgAF = Nrg / (1.0 + np.exp(-kg * (gAF - sigmag)))
    QaAF = Nra / (1.0 + np.exp(-ka * (aAF - sigmaa)))
    QAF = np.sqrt(QgAF * QaAF)
    
    QgBF = Nrg / (1.0 + np.exp(-kg * (gBF - sigmag)))
    QaBF = Nra / (1.0 + np.exp(-ka * (aBF - sigmaa)))
    QBF = np.sqrt(QgBF * QaBF)

    p, q = xrcw.shape
    wtA = np.where(gA >= Td, gA ** Lg, 0.0)
    wtB = np.where(gB >= Td, gB ** Lg, 0.0)

    wt_sum = np.sum(wtA + wtB)
    if wt_sum == 0:
        return 0.0

    # Fusion Artifacts (NABF)
    na = np.where((gF > gA) & (gF > gB), 1.0, 0.0)
    NABF = np.sum(na * ((1.0 - QAF) * wtA + (1.0 - QBF) * wtB)) / wt_sum
    
    return NABF