import numpy as np
from phasepack import phasecong


def correlation_coeff(x, y, C=1e-8):
    x = x.flatten()
    y = y.flatten()

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Dùng N-1 (unbiased) theo công thức paper
    sigma_xy = np.sum((x - mean_x) * (y - mean_y)) / (len(x) - 1)
    sigma_x  = np.std(x, ddof=1)
    sigma_y  = np.std(y, ddof=1)

    return (sigma_xy + C) / (sigma_x * sigma_y + C)


def compute_phase_congruency(img):
    M, m, ori, ft, PC, EO, T = phasecong(img)

    pc          = np.array(PC)
    max_moment  = np.array(M)
    min_moment  = np.array(m)

    return pc, max_moment, min_moment


def max_select_map(A, B):
    return np.where(A >= B, A, B)


def QP(A, B, F, alpha=1, beta=1, gamma=1):
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    F = F.astype(np.float32)

    # Phase congruency cho A, B, F và max-select map S
    pcA, MA, mA = compute_phase_congruency(A)
    pcB, MB, mB = compute_phase_congruency(B)
    pcF, MF, mF = compute_phase_congruency(F)

    S            = max_select_map(A, B)
    pcS, MS, mS  = compute_phase_congruency(S)

    # Phase congruency correlation
    Pp = max(correlation_coeff(pcA, pcF),
             correlation_coeff(pcB, pcF),
             correlation_coeff(pcS, pcF))

    # Maximum moment correlation
    PM = max(correlation_coeff(MA, MF),
             correlation_coeff(MB, MF),
             correlation_coeff(MS, MF))

    # Minimum moment correlation
    Pm = max(correlation_coeff(mA, mF),
             correlation_coeff(mB, mF),
             correlation_coeff(mS, mF))

    return (Pp ** alpha) * (PM ** beta) * (Pm ** gamma)