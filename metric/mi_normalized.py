import numpy as np


def entropy(img):
    hist = np.histogram(img.flatten(), bins=256, range=(0, 256))[0]
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))


def mutual_information(img1, img2):
    joint_hist = np.histogram2d(
        img1.flatten(),
        img2.flatten(),
        bins=256,
        range=[[0, 256], [0, 256]]
    )[0]

    joint_prob = joint_hist / np.sum(joint_hist)

    px = np.sum(joint_prob, axis=1)   # marginal của img1
    py = np.sum(joint_prob, axis=0)   # marginal của img2

    # Vectorized: chỉ tính trên các ô có xác suất > 0
    mask = (joint_prob > 0) & (px[:, None] > 0) & (py[None, :] > 0)

    pxy  = joint_prob[mask]
    pxpy = (px[:, None] * py[None, :])[mask]

    return np.sum(pxy * np.log2(pxy / pxpy))


def QMI(A, B, F):
    HA = entropy(A)
    HB = entropy(B)
    HF = entropy(F)

    MI_AF = mutual_information(A, F)
    MI_BF = mutual_information(B, F)

    return 2 * (MI_AF / (HA + HF) + MI_BF / (HB + HF))