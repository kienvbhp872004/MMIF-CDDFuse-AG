import numpy as np


def histogram_prob(img, bins=256):
    hist = np.histogram(img.ravel(), bins=bins, range=(0,256))[0]
    return hist / hist.sum()


def joint_histogram_prob(img1, img2, bins=256):

    joint_hist = np.histogram2d(
        img1.ravel(),
        img2.ravel(),
        bins=bins,
        range=[[0,256],[0,256]]
    )[0]

    return joint_hist / joint_hist.sum()


def tsallis_entropy(hist, q):

    hist = hist[hist > 0]

    return (1 - np.sum(hist ** q)) / (q - 1)


def tsallis_mutual_information(img1, img2, q, bins=256):

    h1 = histogram_prob(img1, bins)
    h2 = histogram_prob(img2, bins)
    h12 = joint_histogram_prob(img1, img2, bins)

    mi = 0

    for i in range(bins):
        for j in range(bins):

            if h12[i,j] > 0 and h1[i] > 0 and h2[j] > 0:

                mi += (h12[i,j] ** q) * ((h1[i] * h2[j]) ** (q - 1))

    mi = (1 - mi) / (1 - q)

    return mi


def QTE(A, B, F, q=1.5, bins=256):

    I_AF = tsallis_mutual_information(A, F, q, bins)
    I_BF = tsallis_mutual_information(B, F, q, bins)
    I_AB = tsallis_mutual_information(A, B, q, bins)

    hA = histogram_prob(A, bins)
    hB = histogram_prob(B, bins)

    H_A = tsallis_entropy(hA, q)
    H_B = tsallis_entropy(hB, q)

    Q_TE = (I_AF + I_BF) / (H_A + H_B - I_AB)

    return Q_TE