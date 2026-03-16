import numpy as np

def mean_intensity(F):
    """
    Cường độ sáng trung bình (Mean Intensity)
    """
    return np.mean(F.astype(np.float64))
