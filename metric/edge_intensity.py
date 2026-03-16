import numpy as np
import cv2

def edge_intensity(F):
    """
    Edge Intensity (Standard version matching Matlab metricsEdge_intensity.m)
    """
    F = F.astype(np.float64)
    
    # Matlab's fspecial('sobel') returns:
    # [ 1  2  1]
    # [ 0  0  0]
    # [-1 -2 -1]
    # Note: cv2.filter2D correlation kernel is effectively flipped compared to convolution.
    # But for Sobel (anti-symmetric), we can just use the kernel as is with filter2D.
    w = np.array([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1]
    ], dtype=np.float64)
    
    # cv2.filter2D(src, ddepth, kernel, borderType)
    # borderType=cv2.BORDER_REPLICATE matches Matlab's 'replicate'
    gx = cv2.filter2D(F, -1, w, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.filter2D(F, -1, w.T, borderType=cv2.BORDER_REPLICATE)
    
    g = np.sqrt(gx**2 + gy**2)
    
    return np.mean(g)
