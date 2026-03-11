import os
import cv2
import numpy as np
import csv

from metric.qc import QC
from metric.qg import QG
from metric.qm import QM
from metric.qcb import QCB
from metric.qcv import QCV
from metric.qmi import QMI
from metric.qncie import QNCIE
from metric.qp import QP
from metric.qs import QS
from metric.qsf import QSF
from metric.qte import QTE
from metric.qy import QY

from algorithm.wavelet_transform import wavelet_fusion

method = ["CT","MRI","PET","SPECT"]
fused = ["CT-MRI","PET-MRI","SPECT-MRI"]
dataset_dir = r"D:\Workspace\Github\Repo\Image-Fusion\Havard-Medical-Image-Fusion-Datasets-main\Havard-Medical-Image-Fusion-Datasets-main"

ct_dir = os.path.join(dataset_dir,fused[0], method[0])
mri_dir = os.path.join(dataset_dir,fused[0], method[1])

# ===== GIỚI HẠN SỐ ẢNH =====
MAX_IMAGES = 5

images = sorted(os.listdir(ct_dir))[:MAX_IMAGES]

results = []

for img_name in images:

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
    qcb = QCB(ct, mri, fused)
    qcv = QCV(ct, mri, fused)
    qmi = QMI(ct, mri, fused)
    qncie = QNCIE(ct, mri, fused)
    qp = QP(ct, mri, fused)
    qs = QS(ct, mri, fused)
    qsf = QSF(ct, mri, fused)
    qte = QTE(ct, mri, fused)
    qy = QY(ct, mri, fused)

    results.append([
        img_name, qc, qg, qm, qcb, qcv, qmi,
        qncie, qp, qs, qsf, qte, qy
    ])

    print("Done: "+ img_name)


# ===== IN KẾT QUẢ =====

header = [
    "Image", "QC", "QG", "QM", "QCB", "QCV",
    "QMI", "QNCIE", "QP", "QS", "QSF", "QTE", "QY"
]

print("\nResults\n")
print("\t".join(header))

for r in results:
    print("\t".join([r[0]] + [f"{v:.4f}" for v in r[1:]]))


# ===== TÍNH AVERAGE =====

avg_values = np.mean([r[1:] for r in results], axis=0)

print("\nAverage")
for name, val in zip(header[1:], avg_values):
    print(f"{name}: {val:.4f}")


# ===== LƯU CSV =====

csv_path = "fusion_metrics_results.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow(header)

    for r in results:
        writer.writerow(r)

    writer.writerow([])
    writer.writerow(["Average"] + list(avg_values))

print(f"\nSaved CSV to: {csv_path}")