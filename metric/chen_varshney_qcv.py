import numpy as np
import cv2


def sobel_edges(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)


def csf_filter(img):
    """
    CSF (Contrast Sensitivity Function) theo Mannos-Sakrison,
    thực hiện trong miền tần số.
    csf(f) = 2.6 * (0.0192 + 0.114*f) * exp(-(0.114*f)^1.1)
    """
    h, w   = img.shape
    F_img  = np.fft.fft2(img.astype(np.float64))

    # Tần số chuẩn hóa (cycles/ảnh) cho mỗi trục
    u = np.fft.fftfreq(h) * h   # cycles theo chiều dọc
    v = np.fft.fftfreq(w) * w   # cycles theo chiều ngang
    U, V = np.meshgrid(u, v, indexing='ij')

    freq = np.sqrt(U ** 2 + V ** 2)    # tần số xuyên tâm

    # Hàm CSF Mannos-Sakrison
    csf = 2.6 * (0.0192 + 0.114 * freq) * np.exp(-(0.114 * freq) ** 1.1)
    csf /= csf.max() + 1e-8            # chuẩn hóa về [0, 1]

    return np.real(np.fft.ifft2(F_img * csf))


def QCV(A, B, F, window=8):
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    F = F.astype(np.float64)

    GA = sobel_edges(A)
    GB = sobel_edges(B)

    h, w  = A.shape
    num   = 0.0
    den   = 0.0

    for i in range(0, h - window, window):
        for j in range(0, w - window, window):
            Aw  = A[i:i + window, j:j + window]
            Bw  = B[i:i + window, j:j + window]
            Fw  = F[i:i + window, j:j + window]
            GAw = GA[i:i + window, j:j + window]
            GBw = GB[i:i + window, j:j + window]

            salA = np.sum(GAw ** 2)
            salB = np.sum(GBw ** 2)

            # Lọc CSF trên sai số giữa ảnh gốc và ảnh hợp nhất
            DA = np.mean(csf_filter(Aw - Fw) ** 2)
            DB = np.mean(csf_filter(Bw - Fw) ** 2)

            num += salA * DA + salB * DB
            den += salA + salB

    return num / (den + 1e-8)