import cv2
import numpy as np


def sobel_gradients(img):
    img = img.astype(np.float32)

    sx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    g = np.sqrt(sx ** 2 + sy ** 2)
    alpha = np.arctan2(sx, sy)

    return g, alpha


def sigmoid(x, k, x0):
    return 1 / (1 + np.exp(-k * (x - x0)))


def QG(A, B, F, L=1):
    # gradients
    gA, aA = sobel_gradients(A)
    gB, aB = sobel_gradients(B)
    gF, aF = sobel_gradients(F)

    eps = 1e-12

    # relative edge strength
    GAF = np.minimum(gF, gA) / (np.maximum(gF, gA) + eps)
    GBF = np.minimum(gF, gB) / (np.maximum(gF, gB) + eps)

    # orientation preservation
    AAF = 1 - np.abs(aA - aF) / (np.pi / 2)
    ABF = 1 - np.abs(aB - aF) / (np.pi / 2)

    # clamp
    AAF = np.clip(AAF, 0, 1)
    ABF = np.clip(ABF, 0, 1)

    # sigmoid parameters (typical values)
    kg = 15
    tg = 0.5

    ka = 15
    ta = 0.75

    QgAF = sigmoid(GAF, kg, tg)
    QaAF = sigmoid(AAF, ka, ta)

    QgBF = sigmoid(GBF, kg, tg)
    QaBF = sigmoid(ABF, ka, ta)

    QAF = QgAF * QaAF
    QBF = QgBF * QaBF

    # weights
    wA = gA ** L
    wB = gB ** L

    numerator = np.sum(QAF * wA + QBF * wB)
    denominator = np.sum(wA + wB) + eps

    QG = numerator / denominator

    return QG