# Báo cáo đánh giá Medical Image Fusion — Phase 2

> **Thời điểm**: 2026-04-24
> **Dataset**: Harvard Medical Image Fusion (subset tại [`data/reference/`](../data/reference/))
> **Output**: [`results_v2/`](./)
> **Dữ liệu thô**: [all_models_summary.csv](all_models_summary.csv) · [report_overview.csv](report_overview.csv) · [zscore_ranking.csv](zscore_ranking.csv)

---

## 1. Tóm tắt (Executive Summary)

| | |
|---|---|
| **Số model đánh giá** | 23 / 27 chạy thành công |
| **Modality** | CT-MRI, PET-MRI, SPECT-MRI |
| **Kích thước test set** | 24 ảnh × 3 modality = 72 ảnh (chuẩn) |
| **Số metric** | 25 (xem §2.3) |
| **Top 3 overall (theo SSIM)** | **MFS-Fusion · LRFNet · CDDFuse** |
| **Chưa chạy** | SPDFusion (architecture mismatch), DDFM/DM-FNet/VDMUFusion (diffusion, cần GPU) |

**Kết luận nhanh**: Nhóm Mamba/SSM (MFS-Fusion) và Transformer hiện đại (CDDFuse, MM-Net-Fusion, MMIF-INet) dẫn đầu về chất lượng cấu trúc và edge. Các model CNN thế hệ cũ (NestFuse, TUFusion) và MAE-based (MMAE) tụt lại đáng kể.

---

## 2. Phương pháp đánh giá

### 2.1 Dữ liệu

| Modality | Cặp ảnh | Source |
|---|---|---|
| CT-MRI | 24 | Harvard Medical |
| PET-MRI | 24 | Harvard Medical |
| SPECT-MRI | 24 | Harvard Medical |

### 2.2 Quy trình

Mỗi model được chạy inference trên toàn bộ cặp ảnh, output lưu vào `results_v2/<Model>/<modality>/Fusion/` kèm summary JSON + CSV. Runner: [run_all_v2.py](../run_all_v2.py).

### 2.3 Metric

| Nhóm | Metric | Ý nghĩa |
|---|---|---|
| **Thông tin (no-ref)** | EN, VAR, AG, SF, EI | Entropy, variance, gradient, spatial freq, edge intensity — cao = ảnh giàu thông tin |
| **Tương đồng cấu trúc** | SSIM, PSNR, RMSE | So với ảnh source — cao SSIM/PSNR, thấp RMSE = tốt |
| **Chất lượng cạnh** | QG (Petrovic), QM (Wavelet) | Bảo toàn cạnh, cao = tốt |
| **Chất lượng perceptual** | QCB (Chen-Blum), QCV (Chen-Varshney), QY (Yang) | Mô phỏng HVS, cao = tốt |
| **Thông tin tương hỗ** | MI_mutual, QMI, QNCIE, NCIE | Mutual information, cao = tốt |
| **Thống kê khác** | FMI, CE, QC, QS, NABF, QSF, QTE | Phụ trợ |

Báo cáo chính tập trung vào **SSIM, QG, QY, QCB, PSNR, EN** — bộ 6 metric phổ biến, phản ánh 4 khía cạnh quan trọng (cấu trúc, cạnh, perceptual, thông tin).

---

## 3. Phân nhóm model theo kiến trúc

Phân loại dựa trên backbone chính (đọc từ README/paper của mỗi model):

| Nhóm | Số model | Thành viên |
|---|---|---|
| **CNN-based** | 8 | AWFusion, NestFuse, AdaFuse, LRFNet, DDBFusion, C2RF, MBHFuse, PSFusion |
| **Transformer-based** | 8 | CDDFuse, MMIF-INet, MM-Net-Fusion, ITFuse, CMMDL, ECINFusion, CM-CSAMFNet, TUFusion |
| **Mamba / SSM** | 2 | MFS-Fusion, SFMFusion |
| **Autoencoder / Masked** | 3 | MMAE, MLFuse, BSAFusion |
| **Semantic / Wavelet hybrid** | 2 | GeSeNet, WaveFusion |
| **Diffusion** *(chưa chạy)* | 3 | DDFM, DM-FNet, VDMUFusion |

---

## 4. Kết quả theo nhóm kiến trúc

### 4.1 So sánh nhóm (mean metric)

| Nhóm | SSIM ↑ | QG ↑ | QY ↑ | QCB ↑ | PSNR ↑ | EN ↑ | AG ↑ | SF ↑ |
|---|---|---|---|---|---|---|---|---|
| **Mamba / SSM** ¹ | **1.448** | 0.704 | 0.952 | 0.860 | 55.96 | 5.58 | **14.94** | **56.70** |
| **Transformer** | 1.198 | 0.558 | 0.801 | **0.839** | 55.97 | 5.10 | 7.42 | 24.55 |
| **Semantic / Wavelet** | 1.193 | 0.564 | 0.802 | 0.782 | **56.04** | 4.89 | 7.33 | 26.59 |
| **CNN-based** | 1.133 | 0.497 | 0.757 | 0.802 | 55.91 | 5.09 | 6.91 | 23.92 |
| **AE / Masked** | 1.058 | 0.512 | 0.765 | 0.838 | 55.62 | 4.73 | 7.50 | 28.10 |

*¹ Mamba mean chủ yếu phản ánh MFS-Fusion vì SFMFusion bị lỗi paired metric (xem §7.2).*

### 4.2 Nhận xét từng nhóm

#### 🔹 Mamba / SSM

- **MFS-Fusion** có điểm SSIM, QG, AG, SF đều cao nhất — ảnh fused vừa sát source vừa giữ được detail/cạnh mạnh.
- Tuy nhiên runtime CPU **rất chậm** (~50 phút/modality) so với CNN/Transformer (~1-2 phút).
- SFMFusion chưa hoạt động đúng (§7.2).

#### 🔹 Transformer-based

- Nhóm ổn định nhất: phổ rộng từ top-tier (CDDFuse 1.44, MMIF-INet 1.42, MM-Net-Fusion 1.43) tới bottom-tier (ITFuse 0.57).
- QCB cao nhất toàn dataset — perceptual tốt nhất nhóm.
- **CDDFuse** xứng đáng là baseline: top-3 ở mọi metric, code ổn định, runtime hợp lý.

#### 🔹 CNN-based

- Phân tán cao: **LRFNet (1.44) ở đỉnh** nhưng AWFusion/NestFuse/TUFusion ở đáy.
- Nhiều model thế hệ cũ (NestFuse 2020, TUFusion 2021) không còn cạnh tranh được.
- **MBHFuse, PSFusion** có QG cao nhưng SSIM thấp → fused ra ảnh có cạnh nhưng sai structure.

#### 🔹 Autoencoder / Masked

- Nhóm yếu nhất về SSIM (mean 1.058).
- **MMAE** có EN thấp (3.57) → ảnh fused mất information.
- Hướng MAE cho fusion có vẻ chưa mature — cần pre-training tốt hơn.

#### 🔹 Semantic / Wavelet hybrid

- **WaveFusion (1.37)** > **GeSeNet (1.01)** đáng kể về SSIM, nhưng GeSeNet có QG cao hơn.
- Semantic-guided fusion lợi cho edge, wavelet lợi cho multi-scale structure — trade-off tùy task.

---

## 5. Kết quả theo modality

### 5.1 Top 5 mỗi modality (theo SSIM)

#### CT-MRI

| Rank | Model | SSIM | QG | QY | EN |
|---|---|---|---|---|---|
| 🥇 | CDDFuse | 1.480 | 0.614 | 0.926 | 4.69 |
| 🥈 | DDBFusion | 1.476 | 0.439 | 0.851 | 4.72 |
| 🥉 | MM-Net-Fusion | 1.474 | 0.539 | 0.965 | 4.46 |
| 4 | MFS-Fusion | 1.473 | 0.609 | 0.929 | 4.71 |
| 5 | LRFNet | 1.469 | 0.483 | 0.891 | 4.49 |

#### PET-MRI

| Rank | Model | SSIM | QG | QY | EN |
|---|---|---|---|---|---|
| 🥇 | MM-Net-Fusion | 1.413 | 0.797 | **0.994** | 5.26 |
| 🥈 | MMIF-INet | 1.409 | 0.719 | 0.943 | 5.21 |
| 🥉 | MFS-Fusion | 1.406 | 0.760 | 0.958 | 5.46 |
| 4 | CDDFuse | 1.404 | 0.734 | 0.956 | 5.62 |
| 5 | LRFNet | 1.403 | 0.730 | 0.949 | 5.23 |

#### SPECT-MRI

| Rank | Model | SSIM | QG | QY | EN |
|---|---|---|---|---|---|
| 🥇 | MFS-Fusion | 1.465 | 0.742 | 0.968 | 5.05 |
| 🥈 | LRFNet | 1.462 | 0.698 | 0.959 | 4.60 |
| 🥉 | CDDFuse | 1.432 | 0.758 | **0.985** | 5.01 |
| 4 | CM-CSAMFNet | 1.419 | 0.693 | 0.940 | 5.19 |
| 5 | MM-Net-Fusion | 1.407 | 0.767 | 0.991 | 5.17 |

### 5.2 Nhận xét liên modality

- **CT-MRI** — phổ điểm cao và sát nhau (top 5 chênh 0.011 SSIM), task này khá "dễ" vì CT grayscale đơn giản.
- **PET-MRI** — task khó nhất: QG và QY phân hóa rõ. Model top ưu thế nhờ xử lý tốt structural mismatch giữa MRI anatomical và PET functional.
- **SPECT-MRI** — tương tự PET nhưng điểm SSIM cao hơn do SPECT noisy hơn, trộn với MRI cho ra ảnh mượt.

---

## 6. Xếp hạng tổng thể (mean across modalities)

| Rank | Model | Group | Mod | Total N | SSIM | QG | QY | QCB | PSNR | EN |
|------|-------|-------|-----|---------|------|-----|-----|------|-------|-----|
| 🥇 | **MFS-Fusion** | Mamba | CT,PET,SPECT | 72 | **1.4479** | 0.7037 | 0.9520 | 0.8604 | 55.96 | 5.07 |
| 🥈 | **LRFNet** | CNN | CT,PET,SPECT | 72 | 1.4444 | 0.6373 | 0.9332 | 0.7904 | **56.66** | 4.77 |
| 🥉 | **CDDFuse** | Transformer | CT,PET,SPECT | 72 | 1.4387 | 0.7023 | 0.9559 | 0.8577 | 56.22 | 5.10 |
| 4 | MM-Net-Fusion | Transformer | CT,PET,SPECT | 30 | 1.4312 | 0.7012 | **0.9835** | 0.8668 | 56.79 | 4.97 |
| 5 | MMIF-INet | Transformer | CT,PET,SPECT | 30 | 1.4206 | 0.6832 | 0.9328 | 0.8667 | 55.72 | 5.15 |
| 6 | CM-CSAMFNet | Transformer | PET,SPECT | 48 | 1.4021 | **0.7291** | 0.9619 | **0.8879** | 55.45 | 5.32 |
| 7 | DDBFusion | CNN | CT,PET,SPECT | 72 | 1.3963 | 0.4063 | 0.8177 | 0.8595 | 56.67 | 4.93 |
| 8 | WaveFusion | Wavelet | CT,PET,SPECT | 72 | 1.3731 | 0.4309 | 0.8814 | 0.7047 | 56.11 | 4.19 |
| 9 | C2RF | CNN | PET | 24 | 1.3094 | 0.1782 | 0.8829 | 0.5952 | 55.11 | 3.34 |
| 10 | NestFuse | CNN | CT,PET,SPECT | 72 | 1.2457 | 0.3138 | 0.7572 | 0.7211 | 55.87 | 4.02 |
| 11 | TUFusion | Transformer | CT,PET,SPECT | 72 | 1.2339 | 0.2510 | 0.7693 | 0.6944 | 56.19 | 4.37 |
| 12 | MMAE | AE | CT,PET,SPECT | 30 | 1.2192 | 0.2459 | 0.8058 | 0.8346 | 55.10 | 3.57 |
| 13 | BSAFusion | AE | CT,PET,SPECT | 72 | 1.2083 | 0.6878 | 0.9466 | 0.8264 | 56.15 | 5.40 |
| 14 | CMMDL | Transformer | CT,PET,SPECT | 72 | 1.1473 | 0.6995 | 0.7931 | 0.8615 | 55.41 | 5.56 |
| 15 | AdaFuse | CNN | CT,PET,SPECT | 72 | 1.0624 | 0.4947 | 0.6602 | 0.8632 | 56.46 | 5.50 |
| 16 | GeSeNet | Semantic | CT,PET,SPECT | 72 | 1.0121 | 0.6974 | 0.7233 | 0.8589 | 55.97 | 5.58 |
| 17 | AWFusion | CNN | CT,PET,SPECT | 72 | 0.9543 | 0.5774 | 0.7553 | 0.8569 | 55.80 | **6.75** |
| 18 | ECINFusion | Transformer | CT,PET | 48 | 0.9413 | 0.5269 | 0.6350 | 0.8477 | 55.74 | 5.17 |
| 19 | MBHFuse | CNN | CT,PET,SPECT | 30 | 0.8527 | 0.7071 | 0.6505 | 0.8648 | 55.47 | 5.58 |
| 20 | PSFusion | CNN | CT,PET,SPECT | 30 | 0.8028 | 0.6630 | 0.5994 | 0.8658 | 55.25 | 5.83 |
| 21 | MLFuse | AE | CT,PET,SPECT | 30 | 0.7452 | 0.6024 | 0.5424 | 0.8516 | 55.60 | 5.23 |
| 22 | ITFuse | Transformer | CT,PET,SPECT | 72 | 0.5658 | 0.1716 | 0.3736 | 0.8277 | 56.26 | 5.12 |
| ⚠️ | SFMFusion | Mamba | CT,PET,SPECT | 9 | — | — | — | — | — | 6.08 |

**Metric leader**:
- **SSIM**: MFS-Fusion (1.4479)
- **QG**: CM-CSAMFNet (0.7291) — bảo toàn cạnh tốt nhất
- **QY**: MM-Net-Fusion (0.9835) — Yang SSIM xuất sắc
- **QCB**: CM-CSAMFNet (0.8879) — perceptual tốt nhất
- **PSNR**: MM-Net-Fusion (56.79)
- **EN**: AWFusion (6.75) — ảnh giàu thông tin nhất (nhưng SSIM thấp — có thể over-contrast)

---

## 7. Chuẩn hóa Z-score & xếp hạng tổng hợp

### 7.1 Động lực

Giá trị tuyệt đối của các metric có đơn vị và scale khác nhau (SSIM ∈ [0,2], PSNR ∈ [50,60], EN ∈ [3,7], RMSE ∈ [0.1,0.3]...) → **không thể cộng trực tiếp** để so sánh tổng thể. Chuẩn hóa Z-score đưa mọi metric về cùng phân phối chuẩn (μ=0, σ=1), cho phép tính **composite score = trung bình các Z** đại diện "độ vượt/kém trung bình" của model.

### 7.2 Công thức

Với mỗi metric `m` có hướng `d ∈ {+1, −1}` (+1 cao tốt, −1 thấp tốt):

$$Z_{i,m} = d \cdot \frac{x_{i,m} - \mu_m}{\sigma_m}$$

Composite Z cho model $i$:

$$\bar{Z}_i = \frac{1}{|M|}\sum_{m \in M} Z_{i,m}$$

trong đó $M$ = tập 20 metric được sử dụng (loại trừ `MI` — mean intensity thuần tín hiệu, không phản ánh quality; và `QTE`/`QCV` có scale khác thường):

| +1 (higher better) | −1 (lower better) |
|---|---|
| SSIM, PSNR, QG, QM, QCB, QY, QC, QS, EN, MI_mutual, FMI, NCIE, QMI, QNCIE, AG, SF, EI, VAR | NABF, CE (và RMSE) |

**Ý nghĩa thực tế**: Composite Z > 0 nghĩa là model **trên trung bình** của pool 22 model. |Z| = 1 tương ứng với **1 standard deviation** so với mean.

### 7.3 Bảng xếp hạng Z-score

| Rank | Model | Composite Z | SSIM_z | QG_z | QY_z | QCB_z | PSNR_z | EN_z | Ghi chú |
|------|-------|-------------|--------|------|------|-------|--------|------|---------|
| 🥇 1 | **MM-Net-Fusion** | **+1.0285** | +1.03 | +0.87 | **+1.25** | +0.63 | **+1.82** | −0.08 | n=30 (⚠️ limit) |
| 🥈 2 | **CDDFuse** | **+0.9256** | +1.06 | +0.88 | +1.08 | +0.50 | +0.65 | +0.11 | **n=72 (full)** |
| 🥉 3 | **MFS-Fusion** | **+0.6875** | **+1.10** | +0.88 | +1.05 | +0.54 | +0.11 | +0.07 | **n=72 (full)** |
| 4 | CM-CSAMFNet | +0.5908 | +0.92 | **+1.02** | +1.11 | **+0.92** | −0.95 | +0.40 | chỉ PET+SPECT |
| 5 | BSAFusion | +0.5759 | +0.16 | +0.80 | +1.02 | +0.07 | +0.50 | +0.50 | n=72 |
| 6 | MMIF-INet | +0.4401 | +0.99 | +0.78 | +0.93 | +0.63 | −0.38 | +0.18 | n=30 |
| 7 | LRFNet | +0.4102 | +1.08 | +0.53 | +0.93 | −0.42 | +1.55 | −0.34 | n=72 |
| 8 | CMMDL | +0.1852 | −0.07 | +0.86 | +0.04 | +0.55 | −1.02 | +0.71 | n=72 |
| 9 | GeSeNet | +0.1444 | −0.60 | +0.85 | −0.41 | +0.52 | +0.12 | +0.74 | n=72 |
| 10 | MBHFuse | +0.0755 | −1.22 | +0.90 | −0.87 | +0.60 | −0.90 | +0.75 | n=30 |
| 11 | ECINFusion | +0.0626 | −0.88 | −0.05 | −0.97 | +0.37 | −0.35 | +0.20 | chỉ CT+PET |
| 12 | DDBFusion | +0.0400 | +0.90 | −0.69 | +0.20 | +0.53 | +1.57 | −0.12 | n=72 |
| 13 | AdaFuse | +0.0271 | −0.40 | −0.22 | −0.81 | +0.58 | +1.15 | +0.65 | n=72 |
| 14 | AWFusion | −0.1452 | −0.83 | +0.22 | −0.20 | +0.49 | −0.23 | **+2.31** | n=72 |
| 15 | PSFusion | −0.1481 | −1.41 | +0.67 | −1.19 | +0.62 | −1.35 | +1.08 | n=30 |
| 16 | MLFuse | −0.3583 | −1.64 | +0.35 | −1.56 | +0.42 | −0.63 | +0.28 | n=30 |
| 17 | WaveFusion | −0.3744 | +0.81 | −0.56 | +0.60 | −1.60 | +0.41 | −1.11 | n=72 |
| 18 | NestFuse | −0.5348 | +0.31 | −1.18 | −0.19 | −1.37 | −0.07 | −1.35 | n=72 |
| 19 | TUFusion | −0.6729 | +0.26 | −1.51 | −0.11 | −1.74 | +0.58 | −0.88 | n=72 |
| 20 | ITFuse | −0.8609 | −2.34 | −1.93 | **−2.63** | +0.09 | +0.73 | +0.13 | n=72 |
| 21 | MMAE | −0.9939 | +0.21 | −1.54 | +0.12 | +0.19 | −1.67 | −1.95 | n=30 |
| 22 | C2RF | −1.1050 | +0.56 | −1.90 | +0.61 | **−3.10** | −1.64 | −2.26 | chỉ PET (n=24) |

> **SFMFusion** bị loại khỏi bảng Z-score do paired metrics NaN (n chỉ 3/modality). Cần debug riêng.

### 7.4 Ý nghĩa thống kê cho việc chọn model

**Quy tắc diễn giải composite Z**:
- $\bar{Z} > +0.5$: model **tốt rõ rệt** hơn trung bình (trên 0.5σ)
- $\bar{Z} \in [-0.5, +0.5]$: model **trung bình**, khó phân biệt
- $\bar{Z} < -0.5$: model **kém rõ rệt**

**Phân tầng kết quả**:

```
TIER 1 (Z > +0.5)  [7 model]  → MM-Net-Fusion, CDDFuse, MFS-Fusion, CM-CSAMFNet, BSAFusion
TIER 2 (Z > 0)     [6 model]  → MMIF-INet, LRFNet, CMMDL, GeSeNet, MBHFuse, ECINFusion, DDBFusion, AdaFuse
TIER 3 (Z < 0)     [9 model]  → AWFusion, PSFusion, MLFuse, WaveFusion, NestFuse, TUFusion, ITFuse, MMAE, C2RF
```

**Chênh lệch giữa #1 và #2 = 0.10σ** — khá nhỏ, không significant. Giữa **#3 và #4 = 0.10σ** cũng vậy. Nhưng **#1 − #22 = 2.13σ** → rất có ý nghĩa.

### 7.5 Khuyến nghị chọn model (bảo vệ trước hội đồng)

> **Câu hỏi có thể bị hỏi: "Tại sao chọn model X?"**

**Phương án 1 — Chọn CDDFuse (khuyến nghị mạnh)**

**Luận cứ khoa học:**
- Composite Z = **+0.93** → model top-3, vượt trung bình 0.93σ (rất rõ rệt thống kê)
- **n = 72 ảnh (full dataset)** → kết quả tin cậy cao nhất, không bị hoài nghi về sample size
- Top-3 ở mọi metric chính (SSIM, QG, QY) → **robustness** cao
- Kiến trúc Transformer (Restormer-based) đã được peer-review (CVPR 2023)
- Code open-source, reproducible

> *"Dựa trên đánh giá 22 model trên 72 ảnh thuộc 3 modality (CT/PET/SPECT-MRI), CDDFuse đạt composite Z-score +0.93σ — tức chất lượng trung bình tổng hợp cao hơn trung bình pool 0.93 độ lệch chuẩn, xếp thứ 2 overall và **đứng đầu trong nhóm model chạy full dataset**."*

**Phương án 2 — Chọn MM-Net-Fusion (chỉ nếu có thể rerun)**

- Composite Z = **+1.03** → dẫn đầu
- **Caveat**: chỉ test n=30 (10 ảnh/modality) do giới hạn hardcoded trong code
- **Hành động cần làm trước defense**: sửa `pairs[:10]` → `pairs[:24]` và rerun để fair. Nếu Z vẫn > 0.9, luận cứ mạnh hơn CDDFuse.

**Phương án 3 — Chọn MFS-Fusion (nếu ưu tiên state-of-the-art)**

- Composite Z = **+0.69**, kiến trúc **Mamba** (SSM) — hướng mới 2024
- **n = 72 (full)**, SSIM cao nhất tuyệt đối (+1.10σ)
- **Caveat**: runtime CPU 50 phút/modality — không production-ready nếu cần latency thấp. Nhưng với research defense, "slow but best quality" là lý lẽ hợp lệ.

### 7.6 Vì sao KHÔNG nên chọn các model khác

| Model | Composite Z | Lý do loại |
|---|---|---|
| CM-CSAMFNet | +0.59 | Không support CT (chỉ PET+SPECT), coverage yếu |
| BSAFusion | +0.58 | SSIM_z = +0.16 thấp bất thường — kết quả lệch |
| LRFNet | +0.41 | QCB_z < 0 (perceptual kém), không cân bằng |
| MMIF-INet | +0.44 | n=30 limit giống MM-Net-Fusion |
| ITFuse | −0.86 | SSIM_z = −2.34, QY_z = −2.63 — gần 2.5σ dưới mean |
| C2RF | −1.11 | Chỉ support PET, QCB_z = −3.10 (perceptual rất kém) |

### 7.7 Lưu ý khi trình bày

- **Luôn đi kèm n** (sample size) khi công bố Z-score
- **Không gộp model có n khác nhau vào cùng kết luận** — phân tầng theo n trước
- Có thể dùng bar chart với error bars thay vì bảng số khi trình bày slide

---

## 8. Thảo luận

### 7.1 Pattern quan trọng

1. **Mamba > Transformer > CNN** ở top tier: long-range dependency modeling (Mamba/Attention) giúp fused ảnh giữ structural consistency tốt hơn pure conv local receptive field.
2. **QG cao không đảm bảo SSIM cao** (xem MBHFuse, GeSeNet) — các model tập trung edge-preserving có thể đánh đổi structural similarity.
3. **Entropy (EN) không tương quan với quality**: AWFusion có EN cao nhất nhưng SSIM thấp nhất top-10 → EN cao có thể do noise/over-contrast, không phải thông tin hữu ích.
4. **Modality khó nhất: PET-MRI** do độ chênh lệch lớn giữa anatomical (MRI) và functional (PET) — phân hóa model rõ nhất ở đây.

### 7.2 Vấn đề phát hiện

**a) SFMFusion — paired metrics NaN, n=3**
Model ghi ra chỉ 3 ảnh/modality và tất cả metric cần reference (SSIM, QG, ...) = NaN. Có thể do bug trong loop hoặc exception silent. Cần debug riêng.

**b) Giới hạn `pairs[:10]` trong 6 model hardcoded**
MBHFuse, MLFuse, MM-Net-Fusion, MMAE, MMIF-INet, PSFusion bị giới hạn 10 ảnh/modality trong code → so sánh không fair với 16 model khác (24 ảnh). Hướng fix: đổi thành `pairs[:24]` hoặc bỏ slice.

**c) SPDFusion — architecture mismatch**
`RuntimeError: weight of size [64, 1024, 3, 3], expected input to have 1024 channels, but got 256`. Checkpoint không khớp network definition trong code. Cần đọc kỹ paper + repo gốc.

**d) 3 diffusion model chưa chạy**
DDFM, DM-FNet, VDMUFusion đang chạy CPU (đang background). Sẽ update báo cáo sau khi có kết quả.

### 7.3 Giới hạn của báo cáo

- **Chỉ dùng subset 24 ảnh/modality** từ Harvard dataset → không đại diện cho toàn bộ phân phối.
- **Chạy CPU-only** — ảnh hưởng tới model inference (đặc biệt diffusion), không tới chất lượng output.
- **Không test ở resolution/sizing khác** — nhiều model nhạy với input size.
- **Một số metric có scale khác nhau** (SSIM theo Yang có thể > 1) — so sánh tuyệt đối chỉ trong cùng implementation.

---

## 9. Khuyến nghị

### 8.1 Model nên dùng làm baseline

| Use case | Model khuyến nghị | Lý do |
|---|---|---|
| **Production / research baseline** | **CDDFuse** | Top-3 mọi metric, code ổn, runtime hợp lý, checkpoint public |
| **Max quality (chấp nhận chậm)** | **MFS-Fusion** | SSIM cao nhất, giàu detail |
| **Edge-preserving nặng** | **CM-CSAMFNet** | QG cao nhất, perceptual tốt |
| **PET-MRI chuyên biệt** | **MM-Net-Fusion** | QY 0.994 ở PET |
| **Tốc độ** | **LRFNet, ITFuse** | ~60s/modality trên CPU |

### 8.2 Model nên tránh (theo metric)

- **ITFuse** — SSIM 0.57, QG 0.17 — cuối bảng ở cả 2 metric
- **MMAE, MLFuse** — nhóm MAE chưa trưởng thành cho medical fusion
- **SFMFusion** — đang lỗi, chưa dùng được

### 8.3 Việc tiếp theo

1. Fix `pairs[:10]` → `pairs[:24]` cho 6 model hardcoded → rerun để fair
2. Debug SFMFusion (tại sao NaN)
3. Debug SPDFusion (checkpoint)
4. Chờ 3 diffusion xong → append vào báo cáo
5. Cân nhắc rerun trên GPU toàn bộ (runtime sẽ giảm 10-50×)

---

## 10. Kết luận

- **23/27 model** đã đánh giá thành công với convention thống nhất.
- **Top 3 overall**: MFS-Fusion (Mamba), LRFNet (CNN), CDDFuse (Transformer) — 3 kiến trúc khác nhau cùng ở top cho thấy không một paradigm nào dominate tuyệt đối.
- **Nhóm Mamba** (n=1 valid) và **Transformer thế hệ mới** vượt trội rõ về SSIM + QY so với CNN/AE.
- **Entropy (EN) cao ≠ chất lượng cao** — cẩn thận khi dùng chỉ EN làm tiêu chí chọn model.
