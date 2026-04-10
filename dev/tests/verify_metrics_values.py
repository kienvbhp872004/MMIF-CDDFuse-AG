import numpy as np
import cv2
import os
import sys

# Add the project root to sys.path to import the metric package
sys.path.append(os.getcwd())

import metric as m

def generate_test_data(shape=(256, 256)):
    np.random.seed(42)
    A = np.random.randint(0, 256, shape).astype(np.float64)
    B = np.random.randint(0, 256, shape).astype(np.float64)
    F = np.random.randint(0, 256, shape).astype(np.float64)
    return A, B, F

def verify_all():
    A, B, F = generate_test_data()
    print(f"Test Image Shape: {A.shape}")
    print("-" * 50)
    
    results = {}
    
    # 1. Average Gradient
    results['AG'] = m.average_gradient(F)
    
    # 2. Entropy
    results['IE'] = m.entropy(F)
    
    # 3. Spatial Frequency
    results['SF'] = m.spatial_frequency(F)
    
    # 4. Variance
    results['Var'] = m.variance(F)
    
    # 5. Mutual Information
    results['MI'] = m.mutual_information(A, B, F)
    
    # 6. PSNR
    results['PSNR'] = m.psnr(A, B, F)
    
    # 7. SSIM
    results['SSIM'] = m.ssim(A, B, F)
    
    # 8. QG (Qabf)
    results['QG'] = m.qg_petrovic(A, B, F)
    
    # 9. RMSE
    results['RMSE'] = m.rmse(A, B, F)
    
    # 10. Cross Entropy
    results['CE'] = m.cross_entropy(A, B, F)
    
    # 11. FMI
    results['FMI'] = m.fmi(A, B, F)
    
    for k, v in results.items():
        print(f"{k:10}: {v:.8f}")

if __name__ == "__main__":
    verify_all()
