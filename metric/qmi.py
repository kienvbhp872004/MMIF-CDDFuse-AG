import numpy as np


def entropy(img):
    hist = np.histogram(img.flatten(), bins=256, range=(0,256))[0]
    prob = hist / np.sum(hist)

    prob = prob[prob > 0]

    return -np.sum(prob * np.log2(prob))


def mutual_information(img1, img2):

    joint_hist = np.histogram2d(
        img1.flatten(),
        img2.flatten(),
        bins=256,
        range=[[0,256],[0,256]]
    )[0]

    joint_prob = joint_hist / np.sum(joint_hist)

    px = np.sum(joint_prob, axis=1)
    py = np.sum(joint_prob, axis=0)

    mi = 0

    for i in range(256):
        for j in range(256):
            if joint_prob[i,j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint_prob[i,j] * np.log2(
                    joint_prob[i,j] / (px[i]*py[j])
                )

    return mi


def QMI(A, B, F):

    HA = entropy(A)
    HB = entropy(B)
    HF = entropy(F)

    MI_AF = mutual_information(A, F)
    MI_BF = mutual_information(B, F)

    qmi = 2 * (
        (MI_AF / (HA + HF)) +
        (MI_BF / (HB + HF))
    )

    return qmi