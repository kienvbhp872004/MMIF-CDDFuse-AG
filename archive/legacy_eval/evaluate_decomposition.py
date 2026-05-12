"""
Script đánh giá hiệu năng các thuật toán Image Fusion dựa trên Decomposition.
Sử dụng các metric từ folder metric/.

So sánh 4 thuật toán:
1. DWT (Discrete Wavelet Transform)
2. NSST (Non-subsampled Shearlet Transform)
3. Pyramid (Laplacian Pyramid)
4. Guided Filter (GF Decomposition)
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from tabulate import tabulate
import time

# Thêm path để import module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from algorithm.decomposition import DWTFusion, NSSTFusion, PyramidFusion, GuidedFilterFusion
import metric as m

# Danh sách các metric sẽ sử dụng
METRIC_FUNCS = {
    "IE": m.entropy,
    "AG": m.average_gradient,
    "SF": m.spatial_frequency,
    "PSNR": m.psnr,
    "SSIM": m.ssim,
    "QG": m.qg_petrovic,
    "MI": m.mutual_information,
    "VI": m.variance,
}

def evaluate_fusion():
    print("🚀 Khởi động quá trình đánh giá Image Fusion...")
    
    # Cấu hình dataset
    dataset_base = r"d:\Workspace\Github\Repo\Image-Fusion\Havard-Medical-Image-Fusion-Datasets-main\Havard-Medical-Image-Fusion-Datasets-main"
    pair_type = "SPECT-MRI"
    dir_A = os.path.join(dataset_base, pair_type, "SPECT")
    dir_B = os.path.join(dataset_base, pair_type, "MRI")
    
    # Chọn subset ảnh để test (ví dụ 3 cặp đầu tiên)
    image_names = sorted(os.listdir(dir_A))[:3]
    print(f"📂 Đang xử lý dataset: {pair_type} ({len(image_names)} cặp ảnh)")

    # Khởi tạo các fuser với cấu hình tiêu chuẩn
    fusers = {
        "DWT (db2, L3)": DWTFusion(wavelet='db2', level=3),
        "NSST (L3, 8dir)": NSSTFusion(levels=3, directions=[8, 8, 8]),
        "Pyramid (L5)": PyramidFusion(levels=5),
        "Guided Filter": GuidedFilterFusion(radius=4, eps=0.01, n_scales=3)
    }

    results = []

    for img_name in image_names:
        print(f"  🔍 Đang đánh giá cặp: {img_name}")
        path_A = os.path.join(dir_A, img_name)
        path_B = os.path.join(dir_B, img_name)
        
        img_A = cv2.imread(path_A, cv2.IMREAD_GRAYSCALE)
        img_B = cv2.imread(path_B, cv2.IMREAD_GRAYSCALE)
        
        if img_A is None or img_B is None:
            continue

        for algo_name, fuser in fusers.items():
            t0 = time.time()
            img_F = fuser.fuse(img_A, img_B)
            duration = time.time() - t0
            
            # Tính toán metric
            stats = {
                "Algorithm": algo_name,
                "Image": img_name,
                "Time (s)": duration
            }
            
            for m_name, m_func in METRIC_FUNCS.items():
                try:
                    # Các metric có interface khác nhau
                    if m_name in ["IE", "AG", "SF", "VI"]:
                        val = m_func(img_F)
                    else:
                        val = m_func(img_A, img_B, img_F)
                    stats[m_name] = round(float(val), 4)
                except Exception as e:
                    stats[m_name] = np.nan
            
            results.append(stats)

    # Tổng hợp kết quả trung bình
    df = pd.DataFrame(results)
    summary = df.groupby("Algorithm").mean(numeric_only=True).reset_index()
    
    # Sắp xếp theo thứ tự algorithm ban đầu
    summary['Algorithm'] = pd.Categorical(summary['Algorithm'], categories=fusers.keys(), ordered=True)
    summary = summary.sort_values("Algorithm")

    print("\n" + "="*80)
    print("📊 KẾT QUẢ ĐÁNH GIÁ TRUNG BÌNH (AVERAGE METRICS)")
    print("="*80)
    print(tabulate(summary, headers='keys', tablefmt='psql', showindex=False))
    
    # Lưu kết quả
    output_csv = "decomposition_evaluation_results.csv"
    summary.to_csv(output_csv, index=False)
    print(f"\n✅ Đã lưu kết quả chi tiết vào: {output_csv}")

if __name__ == "__main__":
    evaluate_fusion()
