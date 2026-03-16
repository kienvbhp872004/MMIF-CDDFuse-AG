import numpy as np
import os
import sys

# Thêm đường dẫn để có thể import từ folder metric
sys.path.append(os.getcwd())

import metric

def manual_wavelet_fusion(A, B):
    h, w = A.shape
    Ah, Aw = A[:h-(h%2), :w-(w%2)], B[:h-(h%2), :w-(w%2)]
    def dwt_step(img):
        LL = (img[0::2, 0::2] + img[0::2, 1::2] + img[1::2, 0::2] + img[1::2, 1::2]) / 2.0
        LH = (img[0::2, 0::2] + img[0::2, 1::2] - img[1::2, 0::2] - img[1::2, 1::2]) / 2.0
        HL = (img[0::2, 0::2] - img[0::2, 1::2] + img[1::2, 0::2] - img[1::2, 1::2]) / 2.0
        HH = (img[0::2, 0::2] - img[0::2, 1::2] - img[1::2, 0::2] + img[1::2, 1::2]) / 2.0
        return LL, LH, HL, HH
    LL1, LH1, HL1, HH1 = dwt_step(A)
    LL2, LH2, HL2, HH2 = dwt_step(B)
    LL = (LL1 + LL2) / 2.0
    LH = np.where(np.abs(LH1) > np.abs(LH2), LH1, LH2)
    HL = np.where(np.abs(HL1) > np.abs(HL2), HL1, HL2)
    HH = np.where(np.abs(HH1) > np.abs(HH2), HH1, HH2)
    fused = np.zeros_like(A)
    fused[0::2, 0::2] = (LL + LH + HL + HH) / 2.0
    fused[0::2, 1::2] = (LL + LH - HL - HH) / 2.0
    fused[1::2, 0::2] = (LL - LH + HL - HH) / 2.0
    fused[1::2, 1::2] = (LL - LH - HL + HH) / 2.0
    return np.clip(fused, 0, 255)

def run_tests():
    size = (32, 32)
    np.random.seed(42)
    A = np.random.randint(50, 150, size).astype(np.float64)
    B = np.random.randint(100, 200, size).astype(np.float64)
    F = manual_wavelet_fusion(A, B)
    
    report = "# BÁO CÁO TỔNG HỢP 22 METRIC ĐÁNH GIÁ ẢNH HỢP NHẤT\n\n"
    report += "Dựa trên tài liệu Liu 2012 và các yêu cầu bổ sung. Thuật toán sử dụng: **Wavelet Transform**.\n\n"
    
    report += "## 1. Ma trận Kiểm thử (Lượt trích 5x5)\n"
    report += "#### Ảnh A (Nguồn 1):\n```\n" + str(A[:5, :5].astype(int)) + "\n```\n"
    report += "#### Ảnh B (Nguồn 2):\n```\n" + str(B[:5, :5].astype(int)) + "\n```\n"
    report += "#### Ảnh F (Hợp nhất theo Wavelet):\n```\n" + str(F[:5, :5]) + "\n```\n\n"
    
    report += "## 2. Kết quả chi tiết (Full 22 Metrics)\n\n"
    report += "| STT | Tên Chỉ số (Metric) | Kết quả | Miền giá trị (Range) | Phân nhóm |\n"
    report += "| :--- | :--- | :--- | :--- | :--- |\n"
    
    # 1. Thống kê cơ bản
    stats = [
        ("Cường độ sáng TB (Mean)", metric.mean_intensity, (F,), "[0, 255]", "Thống kê"),
        ("Độ tương phản (Variance)", metric.variance, (F,), "[0, inf)", "Thống kê"),
        ("Độ sắc nét (Avg Grad)", metric.average_gradient, (F,), "[0, inf)", "Thống kê"),
        ("Entropy", metric.entropy, (F,), "[0, 8]", "Thống kê"),
    ]
    
    # 2. Truyền thống
    trad = [
        ("PSNR", metric.psnr, (A, B, F), "[0, inf) dB", "Truyền thống"),
        ("SSIM (Standard)", metric.ssim, (A, B, F), "[0, 2]", "Truyền thống"),
        ("RMSE", metric.rmse, (A, B, F), "[0, inf)", "Truyền thống"),
        ("Mutual Information (MI)", metric.mutual_information, (A, B, F), "[0, inf)", "Truyền thống"),
        ("Cross Entropy (CE)", metric.cross_entropy, (A, B, F), "[0, inf)", "Truyền thống"),
        ("Spatial Frequency (SF)", metric.spatial_frequency, (F,), "[0, inf)", "Truyền thống"),
        ("Edge Intensity (EI)", metric.edge_intensity, (F,), "[0, inf)", "Truyền thống"),
    ]
    
    # 3. Liu 2012 / VIFB
    liu = [
        ("Q_G (Petrovic)", metric.qg_petrovic, (A, B, F), "[0, 1]", "Liu 2012"),
        ("Q_CB (Chen-Blum)", metric.chen_blum, (A, B, F), "[0, 1]", "Liu 2012"),
        ("Q_CV (Chen-Varshney)", metric.chen_varshney, (A, B, F), "[0, inf)", "Liu 2012"),
        ("Q_M (Wavelet QM)", metric.wavelet_qm, (A, B, F), "[0, 1]", "Liu 2012"),
        ("Q_C (Piella Structural)", metric.piella_qc, (A, B, F), "[-1, 1]", "Liu 2012"),
        ("Q_S (Piella Saliency)", metric.piella_qs, (A, B, F), "[-1, 1]", "Liu 2012"),
        ("Q_Y (Yang SSIM)", metric.yang_ssim, (A, B, F), "[0, 1]", "Liu 2012"),
        ("NMI (Normalized MI)", metric.mi_normalized, (A, B, F), "[0, 2]", "Liu 2012"),
        ("Q_SF (Relative SF)", metric.sf_relative, (A, B, F), "[0, 1]", "Liu 2012"),
        ("Q_NCIE (NCC Entropy)", metric.ncc_entropy, (A, B, F), "[0, 1]", "Nâng cao"),
        ("Q_TE (Tsallis Entropy)", metric.tsallis_entropy, (A, B, F), "[0, inf)", "Nâng cao"),
    ]
    
    all_metrics = stats + trad + liu
    
    for i, (name, func, args, rnge, group) in enumerate(all_metrics, 1):
        try:
            val = func(*args)
            report += f"| {i} | {name} | **{val:.6f}** | {rnge} | {group} |\n"
        except Exception as e:
            report += f"| {i} | {name} | Lỗi: {str(e)} | {rnge} | {group} |\n"

    with open("d:/Repo/Image-Fusion/metric_test_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("Report generated: d:/Repo/Image-Fusion/metric_test_report.md")

if __name__ == "__main__":
    run_tests()
