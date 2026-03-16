import numpy as np
import cv2


def gaussian_blur(img, sigma=2):
    return cv2.GaussianBlur(img, (0, 0), sigma)


def peli_contrast(img):
    Ik  = gaussian_blur(img, 2)
    Ik1 = gaussian_blur(img, 4)

    # C = (bandpass) / (lowpass)
    C = (Ik - Ik1) / (Ik1 + 1e-8)
    return C


def masked_contrast(C, t=1, h=1, p=1, q=1, Z=1e-6):
    return (t * (C ** p)) / (h * (C ** q) + Z)


def QCB(A, B, F):
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    F = F.astype(np.float64)

    CA = masked_contrast(peli_contrast(A))
    CB = masked_contrast(peli_contrast(B))
    CF = masked_contrast(peli_contrast(F))

    # Saliency map: tỉ lệ đóng góp của A và B
    A_map = (CA ** 2) / (CA ** 2 + CB ** 2 + 1e-8)
    B_map = 1 - A_map

    # Quality map: dùng abs để tránh lỗi dấu khi contrast âm
    absCA = np.abs(CA)
    absCB = np.abs(CB)
    absCF = np.abs(CF)

    QAF = np.where(
        absCA <= absCF,
        absCA / (absCF + 1e-8),
        absCF / (absCA + 1e-8)
    )
    QBF = np.where(
        absCB <= absCF,
        absCB / (absCF + 1e-8),
        absCF / (absCB + 1e-8)
    )

    QGQM = A_map * QAF + B_map * QBF

    return np.mean(QGQM)