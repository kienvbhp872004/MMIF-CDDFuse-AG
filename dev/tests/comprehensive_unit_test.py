import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import metric as m

def run_comprehensive_unit_test():
    # 1. Setup small test matrices
    # Base 5x5 for simple metrics
    np.random.seed(42)
    A5 = np.random.randint(0, 256, (5, 5)).astype(np.float64)
    B5 = np.random.randint(0, 256, (5, 5)).astype(np.float64)
    F5 = (A5 + B5) / 2.0
    
    # 15x15 for window-based metrics (SSIM, FMI, QG)
    A15 = np.random.randint(0, 256, (15, 15)).astype(np.float64)
    B15 = np.random.randint(0, 256, (15, 15)).astype(np.float64)
    F15 = (A15 + B15) / 2.0

    report_lines = []
    report_lines.append("# Comprehensive Metrics Unit Test Report")
    report_lines.append(f"Generated at: {np.datetime64('now')}")
    
    report_lines.append("\n## 1. Small Sample Matrices (5x5)")
    report_lines.append("These values were used for simple metrics (IE, AG, SF, etc.):\n")
    report_lines.append("### Matrix A (5x5 Sample - top left 3x3)")
    report_lines.append("```\n" + str(A5[:3, :3]) + "\n```")
    report_lines.append("### Matrix B (5x5 Sample - top left 3x3)")
    report_lines.append("```\n" + str(B5[:3, :3]) + "\n```")
    report_lines.append("### Fused Matrix F (Average - top left 3x3)")
    report_lines.append("```\n" + str(F5[:3, :3]) + "\n```")

    report_lines.append("\n## 2. Unit Test Results\n")
    report_lines.append("| Metric Name | Matrix Size | Result Value | Category |")
    report_lines.append("| :--- | :--- | :--- | :--- |")

    # Metrics list
    results = []
    
    # Standard / Simple
    results.append(("IE (Entropy)", "5x5", m.entropy(F5), "Information"))
    results.append(("Avg Gradient", "5x5", m.average_gradient(F5), "Visual"))
    results.append(("SF (Spatial Freq)", "5x5", m.spatial_frequency(F5), "Visual"))
    results.append(("Var (Std Dev)", "5x5", m.variance(F5), "Visual"))
    
    # Comparative
    results.append(("Mutual Info", "5x5", m.mutual_information(A5, B5, F5), "Information"))
    results.append(("Cross Entropy", "5x5", m.cross_entropy(A5, B5, F5), "Information"))
    results.append(("PSNR", "5x5", m.psnr(A5, B5, F5), "Comparative"))
    results.append(("RMSE", "5x5", m.rmse(A5, B5, F5), "Comparative"))
    results.append(("SSIM", "15x15", m.ssim(A15, B15, F15), "Comparative"))

    # Advanced / Window-based
    results.append(("FMI", "15x15", m.fmi(A15, B15, F15), "Feature"))
    results.append(("QG (Qabf)", "15x15", m.qg_petrovic(A15, B15, F15), "Edge"))
    results.append(("NCIE", "15x15", m.ncie(A15, B15, F15), "Information"))
    
    # NABF with small matrix often gives 0
    results.append(("NABF (Synthetic)", "15x15", m.nabf(A15, B15, F15), "Artifact"))
    
    # VIFB / Structural
    if hasattr(m, 'piella_qc'):
        results.append(("Piella QC", "15x15", m.piella_qc(A15, B15, F15), "Structural"))
    if hasattr(m, 'piella_qs'):
        results.append(("Piella QS", "15x15", m.piella_qs(A15, B15, F15), "Structural"))
    
    for name, size, val, cat in results:
        report_lines.append(f"| {name} | {size} | {val:,.8f} | {cat} |")

    # 3. Special Verification for NABF with Real Image
    report_lines.append("\n## 3. Real Image Verification (NABF)")
    report_lines.append("Since NABF is often 0 on random noise, we test it on a real medical image pair (SPECT-MRI):\n")
    try:
        import cv2
        imgA = cv2.imread(r"Havard-Medical-Image-Fusion-Datasets-main\Havard-Medical-Image-Fusion-Datasets-main\SPECT-MRI\MRI\25015.png", 0)
        imgB = cv2.imread(r"Havard-Medical-Image-Fusion-Datasets-main\Havard-Medical-Image-Fusion-Datasets-main\SPECT-MRI\SPECT\25015.png", 0)
        imgF = cv2.imread(r"results\CDDFuse\25015.png", 0)
        if imgA is not None:
            nabf_real = m.nabf(imgA, imgB, imgF)
            report_lines.append(f"- **NABF (Real Image 25015.png)**: `{nabf_real:.8f}`")
        else:
            report_lines.append("- *[Warning]: Real images not found for NABF test.*")
    except Exception as e:
        report_lines.append(f"- *[Error]: Could not run real image test: {str(e)}*")

    # Write to report file
    report_path = os.path.join("docs", "reports", "comprehensive_unit_test_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"Report generated successfully at: {report_path}")

if __name__ == "__main__":
    run_comprehensive_unit_test()
