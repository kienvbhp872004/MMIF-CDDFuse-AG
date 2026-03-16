import numpy as np


def spatial_frequency_components(I):
    I = I.astype(np.float32)
    M, N = I.shape
    wd = 1 / np.sqrt(2)

    # Row Frequency (horizontal)
    RF = np.sqrt(
        np.sum((I[:, 1:] - I[:, :-1]) ** 2) / (M * N)
    )

    # Column Frequency (vertical)
    CF = np.sqrt(
        np.sum((I[1:, :] - I[:-1, :]) ** 2) / (M * N)
    )

    # Main Diagonal Frequency
    MDF = np.sqrt(
        wd * np.sum((I[1:, 1:] - I[:-1, :-1]) ** 2) / (M * N)
    )

    # Secondary Diagonal Frequency
    SDF = np.sqrt(
        wd * np.sum((I[1:, :-1] - I[:-1, 1:]) ** 2) / (M * N)
    )

    return RF, CF, MDF, SDF


def spatial_frequency(I):
    RF, CF, MDF, SDF = spatial_frequency_components(I)
    SF = np.sqrt(RF**2 + CF**2 + MDF**2 + SDF**2)
    return SF


def reference_spatial_frequency(A, B):

    A = A.astype(np.float32)
    B = B.astype(np.float32)
    wd = 1 / np.sqrt(2)

    # Horizontal gradient
    gradA_H = A[:, 1:] - A[:, :-1]
    gradB_H = B[:, 1:] - B[:, :-1]
    ref_H = np.maximum(np.abs(gradA_H), np.abs(gradB_H))

    # Vertical gradient
    gradA_V = A[1:, :] - A[:-1, :]
    gradB_V = B[1:, :] - B[:-1, :]
    ref_V = np.maximum(np.abs(gradA_V), np.abs(gradB_V))

    # Main diagonal
    gradA_MD = A[1:, 1:] - A[:-1, :-1]
    gradB_MD = B[1:, 1:] - B[:-1, :-1]
    ref_MD = np.maximum(np.abs(gradA_MD), np.abs(gradB_MD))

    # Secondary diagonal
    gradA_SD = A[1:, :-1] - A[:-1, 1:]
    gradB_SD = B[1:, :-1] - B[:-1, 1:]
    ref_SD = np.maximum(np.abs(gradA_SD), np.abs(gradB_SD))

    M, N = A.shape

    RF_R = np.sqrt(np.sum(ref_H**2) / (M * N))
    CF_R = np.sqrt(np.sum(ref_V**2) / (M * N))
    MDF_R = np.sqrt(wd * np.sum(ref_MD**2) / (M * N))
    SDF_R = np.sqrt(wd * np.sum(ref_SD**2) / (M * N))

    SF_R = np.sqrt(RF_R**2 + CF_R**2 + MDF_R**2 + SDF_R**2)

    return SF_R


def QSF(A, B, F):

    SF_F = spatial_frequency(F)
    SF_R = reference_spatial_frequency(A, B)

    QSF = abs(SF_F - SF_R) / (SF_R + 1e-12)

    return QSF