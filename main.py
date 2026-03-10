import os
import cv2
import numpy as np

from metric.qc import QC
from metric.qg import QG
from metric.qm import QM

from algorithm.wavelet_transform import wavelet_fusion


dataset_dir = r"D:\Workspace\Github\Repo\Image-Fusion\Havard-Medical-Image-Fusion-Datasets-main\Havard-Medical-Image-Fusion-Datasets-main\CT-MRI"

ct_dir = os.path.join(dataset_dir, "CT")
mri_dir = os.path.join(dataset_dir, "MRI")

# ===== GIỚI HẠN SỐ ẢNH =====
MAX_IMAGES = 5

images = sorted(os.listdir(ct_dir))[:MAX_IMAGES]

results = []

for img_name in images:
    print(1)
    ct_path = os.path.join(ct_dir, img_name)
    mri_path = os.path.join(mri_dir, img_name)

    ct = cv2.imread(ct_path, 0)
    mri = cv2.imread(mri_path, 0)

    if ct is None or mri is None:
        continue

    fused = wavelet_fusion(ct, mri)

    qc = QC(ct, mri, fused)
    qg = QG(ct, mri, fused)
    qm = QM(ct, mri, fused)

    results.append([img_name, qc, qg, qm])


print("\nResults\n")
print("Image\tQC\tQG\tQM")

for r in results:
    print(f"{r[0]}\t{r[1]:.4f}\t{r[2]:.4f}\t{r[3]:.4f}")


qc_avg = np.mean([r[1] for r in results])
qg_avg = np.mean([r[2] for r in results])
qm_avg = np.mean([r[3] for r in results])

print("\nAverage")
print(f"QC: {qc_avg:.4f}")
print(f"QG: {qg_avg:.4f}")
print(f"QM: {qm_avg:.4f}")