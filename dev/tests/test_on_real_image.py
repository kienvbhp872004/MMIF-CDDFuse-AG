import cv2
import numpy as np
import metric as m
import os

def run_test():
    base_path = r"Havard-Medical-Image-Fusion-Datasets-main\Havard-Medical-Image-Fusion-Datasets-main\SPECT-MRI"
    imgA_path = os.path.join(base_path, "MRI", "25015.png")
    imgB_path = os.path.join(base_path, "SPECT", "25015.png")
    imgF_path = r"results\CDDFuse\25015.png"
    
    # Read images
    A = cv2.imread(imgA_path, cv2.IMREAD_GRAYSCALE)
    B = cv2.imread(imgB_path, cv2.IMREAD_GRAYSCALE)
    F = cv2.imread(imgF_path, cv2.IMREAD_GRAYSCALE)
    
    if A is None or B is None or F is None:
        print("Error: Could not read images.")
        return

    print(f"Metrics for 25015.png (SPECT-MRI, CDDFuse):")
    print("-" * 50)
    
    metrics = {
        "IE": m.entropy(F),
        "AG": m.average_gradient(F),
        "SF": m.spatial_frequency(F),
        "Var (SD)": m.variance(F),
        "MI": m.mutual_information(A, B, F),
        "PSNR": m.psnr(A, B, F),
        "SSIM": m.ssim(A, B, F),
        "QG": m.qg_petrovic(A, B, F),
        "RMSE": m.rmse(A, B, F),
        "CE": m.cross_entropy(A, B, F)
    }
    
    for k, v in metrics.items():
        print(f"{k:10}: {v:.4f}")

if __name__ == "__main__":
    run_test()
