import numpy as np


def histogram_prob(img, bins=256):
    hist = np.histogram(img.ravel(), bins=bins, range=(0, 256))[0]
    return hist / (hist.sum() + 1e-300)


def joint_histogram_prob(img1, img2, bins=256):
    joint_hist = np.histogram2d(
        img1.ravel(), img2.ravel(),
        bins=bins, range=[[0, 256], [0, 256]]
    )[0]
    return joint_hist / (joint_hist.sum() + 1e-300)


def tsallis_entropy(prob, q):
    prob = prob[prob > 0]
    return (1 - np.sum(prob ** q)) / (q - 1)


def tsallis_mutual_information(img1, img2, q, bins=256):
    """
    I_q(A;F) = 1/(q-1) * [ Σᵢⱼ p_AF(i,j)^q / (p_A(i)^(q-1) * p_F(j)) - 1 ]
    
    Với q < 1: (q-1) < 0, sum_term > 1 nên I_q > 0 (đúng lý thuyết).
    """
    h1  = histogram_prob(img1, bins)   # shape (bins,)
    h2  = histogram_prob(img2, bins)   # shape (bins,)
    h12 = joint_histogram_prob(img1, img2, bins)  # shape (bins, bins)

    # Dùng broadcasting, không cần np.ones
    # h1[:, None] shape (bins, 1), h2[None, :] shape (1, bins)
    mask = (h12 > 0) & (h1[:, None] > 0) & (h2[None, :] > 0)

    p_ij = h12[mask]
    p_i  = h1[:, None].repeat(bins, axis=1)[mask]   # hoặc dùng cách dưới
    p_j  = h2[None, :].repeat(bins, axis=0)[mask]

    # Cách gọn hơn (không repeat, chỉ index sau khi broadcast):
    # i_idx, j_idx = np.where(mask)
    # p_i = h1[i_idx]
    # p_j = h2[j_idx]

    numerator   = p_ij ** q
    denominator = (p_i ** (q - 1)) * p_j  # bỏ 1e-300 vì mask đã lọc

    sum_term = np.sum(numerator / denominator)
    return (sum_term - 1) / (q - 1)   # đúng: dấu giống code gốc ✓


def QTE(A, B, F, q=0.5, bins=256):
    I_AF = tsallis_mutual_information(A, F, q, bins)
    I_BF = tsallis_mutual_information(B, F, q, bins)
    I_AB = tsallis_mutual_information(A, B, q, bins)

    hA = histogram_prob(A, bins)
    hB = histogram_prob(B, bins)

    H_A = tsallis_entropy(hA, q)
    H_B = tsallis_entropy(hB, q)

    denom = H_A + H_B - I_AB
    # Mẫu số có thể âm hoặc rất nhỏ → cần bảo vệ
    Q_TE = (I_AF + I_BF) / (denom + 1e-8) if denom > 1e-8 else 0.0

    return Q_TE