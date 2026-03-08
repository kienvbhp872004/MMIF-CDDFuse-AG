import numpy as np
from skimage.util import view_as_windows

def uiqi(x, y):
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    mx = np.mean(x)
    my = np.mean(y)

    vx = np.var(x)
    vy = np.var(y)

    cov = np.mean((x-mx)*(y-my))

    numerator = 4 * cov * mx * my
    denominator = (vx + vy) * (mx**2 + my**2)

    if denominator == 0:
        return 0
    return numerator / denominator


def covariance(x, y):
    mx = np.mean(x)
    my = np.mean(y)
    return np.mean((x-mx)*(y-my))


def QC(A, B, F, win=8):

    windows_A = view_as_windows(A, (win, win))
    windows_B = view_as_windows(B, (win, win))
    windows_F = view_as_windows(F, (win, win))

    H, W = windows_A.shape[:2]

    qc_total = 0
    count = 0

    for i in range(H):
        for j in range(W):

            a = windows_A[i,j]
            b = windows_B[i,j]
            f = windows_F[i,j]

            sigma_AF = covariance(a,f)
            sigma_BF = covariance(b,f)

            denom = sigma_AF + sigma_BF
            if denom == 0:
                sim = 0
            else:
                sim = sigma_AF / denom

            sim = np.clip(sim,0,1)

            Q_AF = uiqi(a,f)
            Q_BF = uiqi(b,f)

            qc = sim * Q_AF + (1-sim) * Q_BF

            qc_total += qc
            count += 1

    return qc_total / count