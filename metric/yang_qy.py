import numpy as np
import cv2

def local_variance(img):
    return np.var(img)

def matlab_style_ssim_single(img1, img2):
    # Simplified ssim for yang_qy
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    v1 = np.var(img1)
    v2 = np.var(img2)
    v12 = np.mean((img1 - mu1) * (img2 - mu2))
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    return ((2 * mu1 * mu2 + C1) * (2 * v12 + C2)) / ((mu1**2 + mu2**2 + C1) * (v1 + v2 + C2))

def QY(A, B, F, window_size=8):
    h, w = A.shape
    scores = []
    for i in range(0, h-window_size+1, window_size):
        for j in range(0, w-window_size+1, window_size):
            Aw = A[i:i+window_size, j:j+window_size]
            Bw = B[i:i+window_size, j:j+window_size]
            Fw = F[i:i+window_size, j:j+window_size]
            ssim_ab = matlab_style_ssim_single(Aw, Bw)
            ssim_af = matlab_style_ssim_single(Aw, Fw)
            ssim_bf = matlab_style_ssim_single(Bw, Fw)
            if ssim_ab >= 0.75:
                varA = np.var(Aw)
                varB = np.var(Bw)
                lam = varA / (varA + varB) if varA + varB != 0 else 0.5
                Qw = lam * ssim_af + (1-lam) * ssim_bf
            else:
                Qw = max(ssim_af, ssim_bf)
            scores.append(Qw)
    return np.mean(scores) if scores else 0