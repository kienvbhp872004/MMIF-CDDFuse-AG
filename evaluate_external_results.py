import os
import sys
import cv2
import numpy as np
import pandas as pd
from tabulate import tabulate
import time
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add current path to import metrics
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import metric as m

# Comprehensive list of metrics from metric/ folder
METRICS_CONFIG = {
    "IE": (m.entropy, "F"),
    "AG": (m.average_gradient, "F"),
    "SF": (m.spatial_frequency, "F"),
    "EI": (m.edge_intensity, "F"),
    "Var": (m.variance, "F"),
    "Mean": (m.mean_intensity, "F"),
    "MI": (m.mutual_information, "ABF"),
    "PSNR": (m.psnr, "ABF"),
    "SSIM": (m.ssim, "ABF"),
    "RMSE": (m.rmse, "ABF"),
    "QG": (m.qg_petrovic, "ABF"),
    "QM": (m.wavelet_qm, "ABF"),
}

def evaluate_folder(fused_dir, mri_dir, pet_dir, model_name):
    print(f"\n--- Evaluating Model: {model_name} ---")
    fused_images = sorted([f for f in os.listdir(fused_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])
    
    if not fused_images:
        print(f"No fused images found in {fused_dir}")
        return None

    results = []
    for img_name in fused_images:
        path_A = os.path.join(mri_dir, img_name)
        path_B = os.path.join(pet_dir, img_name)
        path_F = os.path.join(fused_dir, img_name)
        
        if not os.path.exists(path_A) or not os.path.exists(path_B):
            continue
            
        img_A = cv2.imread(path_A, cv2.IMREAD_GRAYSCALE)
        img_B = cv2.imread(path_B, cv2.IMREAD_GRAYSCALE)
        img_F = cv2.imread(path_F, cv2.IMREAD_GRAYSCALE)
        
        if img_A is None or img_B is None or img_F is None:
            continue

        stats = {"Model": model_name, "Image": img_name}
        for m_name, (m_func, m_type) in METRICS_CONFIG.items():
            try:
                if m_type == "F":
                    val = m_func(img_F)
                else:
                    val = m_func(img_A, img_B, img_F)
                stats[m_name] = round(float(val), 4)
            except:
                stats[m_name] = np.nan
        results.append(stats)
    
    if not results:
        return None
        
    df = pd.DataFrame(results)
    avg_row = df.mean(numeric_only=True).to_dict()
    avg_row["Model"] = model_name
    avg_row["Image"] = "Average"
    return avg_row

def main():
    mri_dir = r"data/MRI"
    pet_dir = r"data/PET"
    model_results_dirs = {
        "CDDFuse": r"results/CDDFuse",
        "DAF-Net": r"results/DAF-Net"
    }
    
    all_summary = []
    for model_name, folder in model_results_dirs.items():
        if os.path.exists(folder):
            summary = evaluate_folder(folder, mri_dir, pet_dir, model_name)
            if summary:
                all_summary.append(summary)
            else:
                print(f"Skip {model_name}: No results found or evaluation failed.")
    
    if all_summary:
        summary_df = pd.DataFrame(all_summary)
        summary_df = summary_df[["Model"] + list(METRICS_CONFIG.keys())]
        print("\n\n" + "="*80)
        print("FINAL COMPARISON OF MODELS")
        print("="*80)
        print(tabulate(summary_df, headers='keys', tablefmt='psql', showindex=False))
        summary_df.to_csv("external_models_comparison.csv", index=False)
        print("\nResults saved to external_models_comparison.csv")

if __name__ == "__main__":
    main()
