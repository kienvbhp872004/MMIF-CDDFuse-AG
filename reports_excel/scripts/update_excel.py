# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import openpyxl
import shutil
import os

src = r'd:\Workspace\Github\Repo\Image-Fusion\Do_Trung_Kien_20224869_v2.xlsx'
out = r'd:\Workspace\Github\Repo\Image-Fusion\Do_Trung_Kien_20224869_v3.xlsx'

wb = openpyxl.load_workbook(src)
ws = wb.active

# ==============================================================
# Noi dung 2 — Row 51: Tuan 5-6 (viet lai theo thuc te da lam)
# ==============================================================
new_week56 = (
    "Tu\u1ea7n 5\u20136: Tri\u1ec3n khai v\u00e0 \u0111\u00e1nh gi\u00e1 20+ thu\u1eadt to\u00e1n SOTA tr\u00ean Harvard Medical Dataset\n"
    "C\u00e0i \u0111\u1eb7t v\u00e0 ch\u1ea1y to\u00e0n b\u1ed9 20 thu\u1eadt to\u00e1n image fusion SOTA (CDDFuse, SPDFusion, TUFusion, "
    "WaveFusion, VDMUFusion, MMIF-DDFM, BSAFusion, SFMFusion, MFS-Fusion, MM-Net, CM-CSAMFNet, "
    "LRFNet, CMMDL, GeSeNet, NestFuse, PSFusion, MLFuse, ECINFusion, DDBFusion, AdaFuse) tr\u00ean "
    "Harvard Medical Image Dataset g\u1ed3m 72 c\u1eb7p \u1ea3nh \u0111a ph\u01b0\u01a1ng th\u1ee9c (24 PET, 24 SPECT, 24 CT).\n"
    "\u0110\u00e1nh gi\u00e1 \u0111\u1ecbnh l\u01b0\u1ee3ng b\u1eb1ng b\u1ed9 22 metrics t\u1ef1 \u0111\u1ed9ng (SSIM, PSNR, RMSE, EN, MI, AG, SF, EI, NCIE, "
    "FMI, NABF, CE, QG, QM, QC, QS, QCB, QY, QMI, QNCIE, VAR, QTE); t\u1ed5ng h\u1ee3p k\u1ebft qu\u1ea3 theo t\u1eebng "
    "modality v\u00e0 chu\u1ea9n h\u00f3a v\u1ec1 composite z-score \u0111\u1ec3 so s\u00e1nh c\u00f4ng b\u1eb1ng gi\u1eefa c\u00e1c m\u00f4 h\u00ecnh.\n"
    "Ph\u00e2n t\u00edch gap analysis: CDDFuse \u0111\u1ea1t composite z = +0.93, x\u1ebfp h\u1ea1ng #1 trong nh\u00f3m ch\u1ea1y \u0111\u1ee7 "
    "3 modality (n=72 \u1ea3nh), v\u01b0\u1ee3t tr\u1ed9i v\u1ec1 SSIM (z=+1.06), QM (z=+1.93), AG (z=+0.95); tuy nhi\u00ean "
    "c\u00f2n \u0111i\u1ec3m y\u1ebfu r\u00f5 v\u1ec1 NABF (z=\u22120.54, artifact cao) v\u00e0 RMSE/PSNR th\u1ea5p h\u01a1n nh\u00f3m d\u1eabn \u0111\u1ea7u \u2014 "
    "x\u00e1c \u0111\u1ecbnh CDDFuse l\u00e0 baseline t\u1ed1t nh\u1ea5t v\u00e0 ph\u00f9 h\u1ee3p nh\u1ea5t \u0111\u1ec3 ti\u1ebfn h\u00e0nh c\u1ea3i ti\u1ebfn c\u00f3 ki\u1ec3m so\u00e1t."
)

ws.cell(row=51, column=1).value = new_week56

wb.save(out)

# Overwrite original
originals = [
    f for f in os.listdir(r'd:\Workspace\Github\Repo\Image-Fusion')
    if '20224869' in f and 'Bao' not in f and 'tien' not in f
       and 'v2' not in f and 'v3' not in f and f.endswith('.xlsx')
]
for fname in originals:
    dst = os.path.join(r'd:\Workspace\Github\Repo\Image-Fusion', fname)
    shutil.copy2(out, dst)

print("OK")
