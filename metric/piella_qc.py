import numpy as np

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
    h, w = A.shape
    qc_total = 0
    count = 0
    # Manual windowing to avoid skimage.util.view_as_windows
    for i in range(0, h - win + 1, 1): # VIFB usually uses step 1 for sliding or step=win for blocks
        for j in range(0, w - win + 1, 1):
            a = A[i:i+win, j:j+win]
            b = B[i:i+win, j:j+win]
            f = F[i:i+win, j:j+win]
            sigma_AF = covariance(a,f)
            sigma_BF = covariance(b,f)
            denom = sigma_AF + sigma_BF
            sim = sigma_AF / denom if denom != 0 else 0
            sim = np.clip(sim, 0, 1)
            Q_AF = uiqi(a,f)
            Q_BF = uiqi(b,f)
            qc = sim * Q_AF + (1-sim) * Q_BF
            qc_total += qc
            count += 1
    return qc_total / count if count > 0 else 0