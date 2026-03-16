import numpy as np

def variance(F):
    """
    Phương sai (Variance) - Một chỉ số đánh giá độ tương phản.
    """
    return np.var(F.astype(np.float64))
