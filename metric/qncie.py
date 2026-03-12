import numpy as np


def entropy_from_hist(hist, base=256):
    hist = hist.astype(np.float64)
    hist = hist / np.sum(hist)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log(hist) / np.log(base))


def joint_entropy(img1, img2, bins=256, base=256):
    hist2d, _, _ = np.histogram2d(
        img1.ravel(),
        img2.ravel(),
        bins=bins,
        range=[[0, 256], [0, 256]]   # ✅ sửa: bao gồm pixel=255
    )
    pxy = hist2d / np.sum(hist2d)
    pxy = pxy[pxy > 0]
    return -np.sum(pxy * np.log(pxy) / np.log(base))


def ncc_entropy(img1, img2, bins=256):
    Hx  = entropy_from_hist(np.histogram(img1, bins=bins, range=(0, 256))[0])  # ✅ sửa
    Hy  = entropy_from_hist(np.histogram(img2, bins=bins, range=(0, 256))[0])  # ✅ sửa
    Hxy = joint_entropy(img1, img2, bins)
    return Hx + Hy - Hxy


def QNCIE(A, B, F):
    NCC_AB = ncc_entropy(A, B)
    NCC_AF = ncc_entropy(A, F)
    NCC_BF = ncc_entropy(B, F)

    R = np.array([
        [1,      NCC_AB, NCC_AF],
        [NCC_AB, 1,      NCC_BF],
        [NCC_AF, NCC_BF, 1     ]
    ])

    # ✅ eigh đảm bảo eigenvalue thực cho ma trận đối xứng
    eigenvalues = np.linalg.eigh(R)[0]

    p = eigenvalues / 3
    p = p[p > 0]

    return 1 + np.sum(p * np.log(p) / np.log(256))