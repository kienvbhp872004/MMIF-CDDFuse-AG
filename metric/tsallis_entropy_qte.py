import numpy as np


def histogram_prob(img, bins=256):
    hist = np.histogram(img.ravel(), bins=bins, range=(0, 256))[0]
    return hist / (hist.sum() + 1e-300)


def joint_histogram_prob(img1, img2, bins=256):
    joint_hist = np.histogram2d(
        img1.ravel(),
        img2.ravel(),
        bins=bins,
        range=[[0, 256], [0, 256]]
    )[0]
    return joint_hist / (joint_hist.sum() + 1e-300)


def tsallis_entropy(hist, q):
    hist = hist[hist > 0]
    return (1 - np.sum(hist ** q)) / (q - 1)


def tsallis_mutual_information(img1, img2, q, bins=256):
    """
    Công thức:
        I_q(A,F) = 1/(1-q) * sum_{i,j} [ h_AF(i,j)^q / (h_F(j) * h_A(i)^(q-1)) ]
    Tương đương:
        I_q = (sum_term - 1) / (q - 1)   [vì 1/(1-q) = -1/(q-1)]
    """
    h1  = histogram_prob(img1, bins)          # p_A(i),  shape (256,)
    h2  = histogram_prob(img2, bins)          # p_F(j),  shape (256,)
    h12 = joint_histogram_prob(img1, img2, bins)  # p_AF(i,j), shape (256,256)

    # Vectorized: chỉ tính trên các ô có xác suất > 0
    mask = (h12 > 0) & (h1[:, None] > 0) & (h2[None, :] > 0)

    p_ij  = h12[mask]                       # h_AF(i,j)
    p_i   = h1[:, None] * np.ones((bins, bins))  # h_A(i)
    p_j   = h2[None, :] * np.ones((bins, bins))  # h_F(j)

    p_i = p_i[mask]
    p_j = p_j[mask]

    # I_q = 1/(1-q) * Σ [ p_ij^q / (p_j * p_i^(q-1)) ]
    numerator   = p_ij ** q
    denominator = p_j * (p_i ** (q - 1)) + 1e-300

    sum_term = np.sum(numerator / denominator)

    return (sum_term - 1) / (q - 1)


def QTE(A, B, F, q=0.5, bins=256):
    I_AF = tsallis_mutual_information(A, F, q, bins)
    I_BF = tsallis_mutual_information(B, F, q, bins)
    I_AB = tsallis_mutual_information(A, B, q, bins)

    hA = histogram_prob(A, bins)
    hB = histogram_prob(B, bins)

    H_A = tsallis_entropy(hA, q)
    H_B = tsallis_entropy(hB, q)

    Q_TE = (I_AF + I_BF) / (H_A + H_B - I_AB + 1e-8)

    return Q_TE