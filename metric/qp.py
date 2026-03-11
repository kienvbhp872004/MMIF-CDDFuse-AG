import numpy as np
from phasepack import phasecong

def correlation_coeff(x, y, C=1e-8):
    x = x.flatten()
    y = y.flatten()

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    sigma_xy = np.mean((x - mean_x) * (y - mean_y))
    sigma_x = np.var(x)
    sigma_y = np.var(y)

    return (sigma_xy + C) / (sigma_x * sigma_y + C)


def compute_phase_congruency(img):
    M, m, ori, ft, PC, EO, T = phasecong(img)

    pc = np.array(PC)
    max_moment = np.array(M)
    min_moment = np.array(m)

    return pc, max_moment, min_moment

def max_select_map(A, B):
    S = np.where(A >= B, A, B)
    return S

def QP(A, B, F, alpha=1, beta=1, gamma=1):

    A = A.astype(np.float32)
    B = B.astype(np.float32)
    F = F.astype(np.float32)

    # phase congruency
    pcA, MA, mA = compute_phase_congruency(A)
    pcB, MB, mB = compute_phase_congruency(B)
    pcF, MF, mF = compute_phase_congruency(F)

    # max select map
    S = max_select_map(A, B)
    pcS, MS, mS = compute_phase_congruency(S)

    # phase congruency correlation
    Cp_AF = correlation_coeff(pcA, pcF)
    Cp_BF = correlation_coeff(pcB, pcF)
    Cp_SF = correlation_coeff(pcS, pcF)

    Pp = max(Cp_AF, Cp_BF, Cp_SF)

    # max moment correlation
    CM_AF = correlation_coeff(MA, MF)
    CM_BF = correlation_coeff(MB, MF)
    CM_SF = correlation_coeff(MS, MF)

    PM = max(CM_AF, CM_BF, CM_SF)

    # min moment correlation
    Cm_AF = correlation_coeff(mA, mF)
    Cm_BF = correlation_coeff(mB, mF)
    Cm_SF = correlation_coeff(mS, mF)

    Pm = max(Cm_AF, Cm_BF, Cm_SF)

    # final metric
    QP = (Pp**alpha) * (PM**beta) * (Pm**gamma)

    return QP

