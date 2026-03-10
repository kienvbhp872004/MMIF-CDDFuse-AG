import cv2
import numpy as np
import pywt

def wavelet_fusion(img1, img2, wavelet='haar'):
    # resize nếu kích thước khác nhau
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # wavelet decomposition
    coeffs1 = pywt.dwt2(img1, wavelet)
    coeffs2 = pywt.dwt2(img2, wavelet)

    LL1, (LH1, HL1, HH1) = coeffs1
    LL2, (LH2, HL2, HH2) = coeffs2

    # fusion rule
    LL = (LL1 + LL2) / 2
    LH = np.maximum(np.abs(LH1), np.abs(LH2))
    HL = np.maximum(np.abs(HL1), np.abs(HL2))
    HH = np.maximum(np.abs(HH1), np.abs(HH2))

    # reconstruct
    fused = pywt.idwt2((LL, (LH, HL, HH)), wavelet)

    # chuẩn hóa
    fused = np.clip(fused, 0, 255).astype(np.uint8)

    return fused