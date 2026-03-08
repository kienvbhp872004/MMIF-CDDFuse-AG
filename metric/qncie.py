import numpy as np


def entropy_from_hist(hist, base=256):
    hist = hist.astype(np.float64)
    hist = hist / np.sum(hist)

    hist = hist[hist > 0]  # tránh log(0)
    return -np.sum(hist * np.log(hist) / np.log(base))


def joint_entropy(img1, img2, bins=256, base=256):
    hist2d, _, _ = np.histogram2d(
        img1.ravel(),
        img2.ravel(),
        bins=bins,
        range=[[0, bins - 1], [0, bins - 1]]
    )

    pxy = hist2d / np.sum(hist2d)
    pxy = pxy[pxy > 0]

    return -np.sum(pxy * np.log(pxy) / np.log(base))


def ncc_entropy(img1, img2, bins=256):
    Hx = entropy_from_hist(np.histogram(img1, bins=bins, range=(0, bins - 1))[0])
    Hy = entropy_from_hist(np.histogram(img2, bins=bins, range=(0, bins - 1))[0])
    Hxy = joint_entropy(img1, img2, bins)

    return Hx + Hy - Hxy


def QNCIE(A, B, F):
    # tính NCC
    NCC_AB = ncc_entropy(A, B)
    NCC_AF = ncc_entropy(A, F)
    NCC_BF = ncc_entropy(B, F)

    # ma trận tương quan
    R = np.array([
        [1, NCC_AB, NCC_AF],
        [NCC_AB, 1, NCC_BF],
        [NCC_AF, NCC_BF, 1]
    ])

    # eigenvalues
    eigenvalues = np.linalg.eigvals(R)

    # chuẩn hóa
    p = eigenvalues / 3

    # tránh log(0)
    p = p[p > 0]

    # entropy
    Q = 1 + np.sum(p * np.log(p) / np.log(256))

    return Q.real