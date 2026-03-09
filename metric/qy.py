import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


def local_variance(img):
    return np.var(img)


def QY(A, B, F, window_size=8):

    h, w = A.shape
    scores = []

    for i in range(0, h-window_size, window_size):
        for j in range(0, w-window_size, window_size):

            Aw = A[i:i+window_size, j:j+window_size]
            Bw = B[i:i+window_size, j:j+window_size]
            Fw = F[i:i+window_size, j:j+window_size]

            ssim_ab = ssim(Aw, Bw)
            ssim_af = ssim(Aw, Fw)
            ssim_bf = ssim(Bw, Fw)

            if ssim_ab >= 0.75:
                varA = local_variance(Aw)
                varB = local_variance(Bw)

                if varA + varB == 0:
                    lam = 0.5
                else:
                    lam = varA / (varA + varB)

                Qw = lam * ssim_af + (1-lam) * ssim_bf
            else:
                Qw = max(ssim_af, ssim_bf)

            scores.append(Qw)

    return np.mean(scores)