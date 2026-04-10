import cv2
import numpy as np

def sobel_gradients(img):
    img = img.astype(np.float64)
    
    # Custom Sobel kernels from metricsQabf.m
    # h3 = [-1 0 1; -2 0 2; -1 0 1] -> Vertical edges (x-gradient)
    # h1 = [1 2 1; 0 0 0; -1 -2 -1] -> Horizontal edges (y-gradient)
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    
    # Using cv2.filter2D to match Matlab's imfilter with these kernels
    # Note: cv2.filter2D(..., borderType=cv2.BORDER_CONSTANT) to match Matlab's default 0 padding
    sx = cv2.filter2D(img, -1, h3, borderType=cv2.BORDER_CONSTANT)
    sy = cv2.filter2D(img, -1, h1, borderType=cv2.BORDER_CONSTANT)
    
    g = np.sqrt(sx**2 + sy**2)
    
    # Orientation matching Matlab's a = atan(sy/sx) with handling for sx=0
    alpha = np.zeros_like(sx)
    mask_sx_nonzero = (sx != 0)
    alpha[mask_sx_nonzero] = np.arctan(sy[mask_sx_nonzero] / sx[mask_sx_nonzero])
    alpha[~mask_sx_nonzero] = np.pi / 2
    
    return g, alpha

def QG(A, B, F):
    """
    Q^AB/F (Gradient Preservation) - Matches Matlab metricsQabf.m exactly.
    """
    # Parameters from Matlab code
    Tg = 0.9994; kg = -15.0; Dg = 0.5
    Ta = 0.9879; ka = -22.0; Da = 0.8    

    gA, aA = sobel_gradients(A)
    gB, aB = sobel_gradients(B)
    gF, aF = sobel_gradients(F)

    # Relative edge strength (GAF, GBF)
    # Matlab logic:
    # if (gA > gF) GAF = gF/gA;
    # else if (gA == gF) GAF = gF; (Wait, this is the weird part in Matlab)
    # else GAF = gA/gF;
    
    def calculate_relative_strength(gsource, gfused):
        GAF = np.zeros_like(gsource)
        
        # Case: gsource > gfused
        mask1 = (gsource > gfused)
        GAF[mask1] = gfused[mask1] / (gsource[mask1] + 1e-12)
        
        # Case: gsource == gfused
        mask2 = (gsource == gfused)
        GAF[mask2] = gfused[mask2] # Matching the weird Matlab logic
        
        # Case: gsource < gfused
        mask3 = (gsource < gfused)
        GAF[mask3] = gsource[mask3] / (gfused[mask3] + 1e-12)
        
        return GAF

    GAF = calculate_relative_strength(gA, gF)
    GBF = calculate_relative_strength(gB, gF)

    # Orientation preservation
    AAF = 1.0 - np.abs(aA - aF) / (np.pi / 2.0)
    ABF = 1.0 - np.abs(aB - aF) / (np.pi / 2.0)

    # Sigmoid functions
    QgAF = Tg / (1.0 + np.exp(kg * (GAF - Dg)))
    QaAF = Ta / (1.0 + np.exp(ka * (AAF - Da)))
    QAF = QgAF * QaAF

    QgBF = Tg / (1.0 + np.exp(kg * (GBF - Dg)))
    QaBF = Ta / (1.0 + np.exp(ka * (ABF - Da)))
    QBF = QgBF * QaBF

    # Total QG
    deno = np.sum(gA + gB)
    nume = np.sum(QAF * gA + QBF * gB)
    
    if deno == 0:
        return 0.0
        
    return nume / deno