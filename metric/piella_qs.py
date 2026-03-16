import numpy as np
import cv2


def uiqi(x, y, eps=1e-10):
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    var_x = np.var(x)
    var_y = np.var(y)

    cov_xy = np.mean((x - mean_x) * (y - mean_y))

    numerator = 4 * cov_xy * mean_x * mean_y
    denominator = (var_x + var_y) * (mean_x**2 + mean_y**2) + eps

    return numerator / denominator


def QS(A, B, F, window_size=8):

    A = A.astype(np.float64)
    B = B.astype(np.float64)
    F = F.astype(np.float64)

    h, w = A.shape
    step = window_size

    qs_sum = 0
    count = 0

    for i in range(0, h - window_size + 1, step):
        for j in range(0, w - window_size + 1, step):

            Aw = A[i:i+window_size, j:j+window_size]
            Bw = B[i:i+window_size, j:j+window_size]
            Fw = F[i:i+window_size, j:j+window_size]

            varA = np.var(Aw)
            varB = np.var(Bw)

            if varA + varB == 0:
                lam = 0.5
            else:
                lam = varA / (varA + varB)

            qAF = uiqi(Aw, Fw)
            qBF = uiqi(Bw, Fw)

            qs_local = lam * qAF + (1 - lam) * qBF

            qs_sum += qs_local
            count += 1

    return qs_sum / count
