import os
import sys
import cv2
import numpy as np
import pandas as pd
from tabulate import tabulate
import time
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add current path to import metrics
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import metric as m

# Comprehensive list of metrics from metric/ folder
# Organized to process faster ones first
METRICS_CONFIG = {
    # F-only (fast)
    "Mean": (m.mean_intensity, "F"),
    "Var": (m.variance, "F"),
    "AG": (m.average_gradient, "F"),
    "IE": (m.entropy, "F"),
    "SF": (m.spatial_frequency, "F"),
    "EI": (m.edge_intensity, "F"),
    
    # Reference-based (varying speeds)
    "MI": (m.mutual_information, "ABF"),
    "CE": (m.cross_entropy, "ABF"),
    "PSNR": (m.psnr, "ABF"),
    "SSIM": (m.ssim, "ABF"),
    "RMSE": (m.rmse, "ABF"),
    "QG": (m.qg_petrovic, "ABF"),
    "QM": (m.wavelet_qm, "ABF"),
    "QCB": (m.chen_blum, "ABF"),
    "QCV": (m.chen_varshney, "ABF"),
    "QY": (m.yang_ssim, "ABF"),
    "QMI": (m.mi_normalized, "ABF"),
    "QSF": (m.sf_relative, "ABF"),
    "QNCIE": (m.ncc_entropy, "ABF"),
    "QTE": (m.tsallis_entropy, "ABF"),
    
    # Slow metrics (have double loops in Python)
    "QC": (m.piella_qc, "ABF"),
    "QS": (m.piella_qs, "ABF")
}

# Optional metrics
if hasattr(m, 'phase_congruency') and m.phase_congruency is not None:
    METRICS_CONFIG["QP"] = (m.phase_congruency, "ABF")

def evaluate_comprehensive():
    print("--- Starting Comprehensive Image Fusion Evaluation ---", flush=True)
    print(f"Total metrics to compute: {len(METRICS_CONFIG)}", flush=True)
    
    dir_MRI = r"d:\Workspace\Github\Repo\Image-Fusion\data\MRI"
    dir_PET = r"d:\Workspace\Github\Repo\Image-Fusion\data\PET"
    dir_Fused = r"d:\Workspace\Github\Repo\Image-Fusion\data\MRI_PET"
    
    output_csv = "fusion_evaluation_comprehensive.csv"

    # Get image files
    fused_images = sorted([f for f in os.listdir(dir_Fused) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])
    
    if not fused_images:
        print("Error: No fused images found.")
        return

    print(f"Dataset size: {len(fused_images)} image pairs.", flush=True)

    results = []
    total_imgs = len(fused_images)

    for idx, img_name in enumerate(fused_images):
        print(f"[{idx+1}/{total_imgs}] Processing: {img_name}...", end="", flush=True)
        img_start_t = time.time()
        
        path_A = os.path.join(dir_MRI, img_name)
        path_B = os.path.join(dir_PET, img_name)
        path_F = os.path.join(dir_Fused, img_name)
        
        if not os.path.exists(path_A) or not os.path.exists(path_B):
            print(" Missing source.", flush=True)
            continue
            
        img_A = cv2.imread(path_A, cv2.IMREAD_GRAYSCALE)
        img_B = cv2.imread(path_B, cv2.IMREAD_GRAYSCALE)
        img_F = cv2.imread(path_F, cv2.IMREAD_GRAYSCALE)
        
        if img_A is None or img_B is None or img_F is None:
            print(" Load error.", flush=True)
            continue

        stats = {"Image": img_name}
        
        for m_name, (m_func, m_type) in METRICS_CONFIG.items():
            try:
                if m_type == "F":
                    val = m_func(img_F)
                else:
                    val = m_func(img_A, img_B, img_F)
                stats[m_name] = round(float(val), 4)
            except Exception:
                stats[m_name] = np.nan
        
        results.append(stats)
        duration = time.time() - img_start_t
        print(f" Done ({duration:.1f}s)", flush=True)

    if not results:
        print("No results generated.")
        return

    df = pd.DataFrame(results)
    
    # Calculate Average
    mean_row = df.mean(numeric_only=True).to_dict()
    mean_row["Image"] = "Average"
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    # Save to CSV
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*100)
    print("ALL METRICS EVALUATION COMPLETE (AVERAGE RESULTS)")
    print("="*100)
    
    summary_df = df.tail(1)
    cols = list(METRICS_CONFIG.keys())
    chunk_size = 7 # Adjusting to fit console screen
    for i in range(0, len(cols), chunk_size):
        subset = ["Image"] + cols[i:i+chunk_size]
        print(tabulate(summary_df[subset], headers='keys', tablefmt='psql', showindex=False))
        print("")
    
    print(f"\n✅ Detailed results for all {len(METRICS_CONFIG)} metrics saved to: {output_csv}")

if __name__ == "__main__":
    evaluate_comprehensive()
