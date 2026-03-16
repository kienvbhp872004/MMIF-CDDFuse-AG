import numpy as np
import cv2

def matlab_style_ssim(img1, img2):
    """
    Manual implementation of SSIM matching Matlab's metricsSsim.m logic.
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    K1 = 0.01
    K2 = 0.03
    L = 255
    C1 = (K1 * L)**2
    C2 = (K2 * L)**2
    
    # 11x11 Gaussian window with sigma 1.5
    # cv2.getGaussianKernel returns a 1D kernel.
    # We create a 2D kernel.
    kernel_size = 11
    sigma = 1.5
    win = cv2.getGaussianKernel(kernel_size, sigma)
    win = np.outer(win, win)
    win = win / np.sum(win)
    
    # Matlab filter2(w, img, 'valid')
    # cv2.filter2D with BORDER_CONSTANT and padding might work, 
    # but filter2 'valid' reduces size.
    # Logic: 
    mu1 = cv2.filter2D(img1, -1, win, borderType=cv2.BORDER_CONSTANT)
    mu2 = cv2.filter2D(img2, -1, win, borderType=cv2.BORDER_CONSTANT)
    
    # To mimic 'valid', we crop the borders
    pad = kernel_size // 2
    mu1 = mu1[pad:-pad, pad:-pad]
    mu2 = mu2[pad:-pad, pad:-pad]
    img1_c = img1[pad:-pad, pad:-pad] # Not quite right for variances
    
    # Instead, let's use the full correlation logic for variances too
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(img1**2, -1, win, borderType=cv2.BORDER_CONSTANT)[pad:-pad, pad:-pad] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, win, borderType=cv2.BORDER_CONSTANT)[pad:-pad, pad:-pad] - mu2_sq
    sigma12 = cv2.filter2D(img1*img2, -1, win, borderType=cv2.BORDER_CONSTANT)[pad:-pad, pad:-pad] - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)

def ssim(A, B, F):
    """
    SSIM (Standard version matching Matlab metricsSsim.m)
    Returns sum of SSIM(A, F) and SSIM(B, F)
    """
    ssim_af = matlab_style_ssim(A, F)
    ssim_bf = matlab_style_ssim(B, F)
    
    return ssim_af + ssim_bf
