import numpy as np
import pywt


def wavelet_decompose(img, levels=2):
    """
    Haar wavelet decomposition
    """
    coeffs = pywt.wavedec2(img, 'haar', level=levels)
    return coeffs


def high_frequency_energy(LH, HL, HH):
    return LH**2 + HL**2 + HH**2


def edge_preservation(A_band, F_band):
    """
    exp(|A - F|)
    """
    return np.exp(-np.abs(A_band - F_band))


def QM(A, B, F, levels=2, beta=None):

    A = A.astype(np.float32)
    B = B.astype(np.float32)
    F = F.astype(np.float32)

    if beta is None:
        beta = [1] * levels

    coeffA = wavelet_decompose(A, levels)
    coeffB = wavelet_decompose(B, levels)
    coeffF = wavelet_decompose(F, levels)

    Q_scales = []

    for s in range(1, levels + 1):

        LH_A, HL_A, HH_A = coeffA[s]
        LH_B, HL_B, HH_B = coeffB[s]
        LH_F, HL_F, HH_F = coeffF[s]

        # Edge preservation
        EP_AF = (
            edge_preservation(LH_A, LH_F) +
            edge_preservation(HL_A, HL_F) +
            edge_preservation(HH_A, HH_F)
        ) / 3

        EP_BF = (
            edge_preservation(LH_B, LH_F) +
            edge_preservation(HL_B, HL_F) +
            edge_preservation(HH_B, HH_F)
        ) / 3

        # weights
        wA = high_frequency_energy(LH_A, HL_A, HH_A)
        wB = high_frequency_energy(LH_B, HL_B, HH_B)

        numerator = np.sum(EP_AF * wA + EP_BF * wB)
        denominator = np.sum(wA + wB) + 1e-12

        Qs = numerator / denominator

        Q_scales.append(Qs)

    # combine scales
    QM = 1
    for s in range(levels):
        QM *= Q_scales[s] ** beta[s]

    return QM