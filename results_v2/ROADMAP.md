# ROADMAP — Cải tiến CDDFuse cho ĐATN

> Master plan thí nghiệm. **Đọc cùng** `PROGRESS.md` (log thực tế) và `CDDFuse_analysis.md` (lý thuyết).
>
> Triết lý: **so sánh có kiểm soát** — mỗi module giữ nguyên CDDFuse, chỉ thay 1 thành phần, test 3-4 alternatives, chọn winner bằng Wilcoxon + Holm correction. KHÔNG combine nhiều thay đổi cùng lúc.

---

## Mục tiêu định lượng (target để vượt baseline)

Theo `fusion-gap-analysis` (xem `CDDFuse_analysis.md` + `zscore_ranking.csv`):

| Metric | Baseline z | Top z | Gap | Target sau cải tiến |
|---|---|---|---|---|
| **NABF** ↓ | -0.54 | +1.19 (MM-Net) | **1.73σ** | z ≥ 0.0 |
| RMSE ↓ | +0.50 | +1.75 (MM-Net) | 1.25σ | z ≥ 1.0 |
| PSNR ↑ | +0.65 | +1.82 (MM-Net) | 1.17σ | z ≥ 1.0 |
| QM ↑ | +1.93 | +2.94 (BSAFusion) | 1.01σ | z ≥ 2.3 |
| **Composite z** | +0.93 | +1.03 (MM-Net*) | 0.10 | **z ≥ 1.05** (vượt MM-Net khi cùng n=24) |

\* MM-Net chỉ chạy n=10/modal — caveat: top model thực tế cùng setting là MFS-Fusion (z=0.69), gap thực = 0.24.

---

## Cấu trúc 3 module × 4 alternatives (chính thức)

### Module A — Fusion Rule

**Vị trí**: `BaseFuseLayer(A + B)` và `DetailFuseLayer(A + B)` trong `evaluate_cddfuse.py:80-81` và paper `train.py`.

**Hypothesis**: Phép `+` thuần là hard rule, không content-aware → tạo artifact (NABF cao). Soft fusion sẽ giảm artifact.

**Câu hỏi nghiên cứu**: *Trong 4 fusion rule, cái nào tốt nhất cho medical fusion?*

| # | Variant | Cơ chế | Params mới | Status | Notes |
|---|---|---|---|---|---|
| A.1 | `FuseRule-Sum` | `A + B` (baseline) | 0 | ✓ baseline pretrained | Reference, không cần train |
| A.2 | `FuseRule-Gated` | `g·A + (1-g)·B`, g từ Conv1×1 | ~16K | ✓ prototype CPU 10 ep | NABF cải thiện 7.3× SPECT, SSIM giảm nhẹ |
| A.3 | `FuseRule-CrossAttn` | `B = CrossAttention(A, B)` | ~50K | pending | Học mutual conditioning |
| A.4 | `FuseRule-ChannelMoE` | Per-channel router chọn A/B/mix | ~10K | pending | Gating ở channel level |

**Acceptance trong module**: variant nào có Wilcoxon NABF p_adj < 0.05 + Cliff's δ ≥ 0.33 + không regression SSIM > 0.5σ → KEEP.

---

### Module B — Pixel-level Loss Selection

**Vị trí**: `L_int^II` trong `utils/loss.py::Fusionloss` — paper dùng `max(I_ir, I_vis)` làm target (line 38: `x_in_max = torch.max(image_y, image_ir)`).

**Hypothesis**: `max` là hard pixel-wise selection → tại boundary 2 modality, target dao động → output nhiễu, RMSE/PSNR kém.

**Câu hỏi nghiên cứu**: *Hard rule vs soft rule vs learnable rule — cái nào win cho medical?*

| # | Variant | Cơ chế | Params mới | Effort |
|---|---|---|---|---|
| B.1 | `PixelSelect-Max` | `max(I_ir, I_vis)` (baseline) | 0 | reference |
| B.2 | `PixelSelect-Mean` | `(I_ir + I_vis)/2` | 0 | dễ (loss-only change) |
| B.3 | `PixelSelect-Saliency` | weight theo `\|∇I\|` magnitude | 0 | dễ (deterministic) |
| B.4 | `PixelSelect-Learnable` | weight map từ small CNN | ~2K | medium |

**Lưu ý**: B.2/B.3 không cần train extra params (chỉ thay loss formulation) → re-train chỉ fuse layers vẫn ổn → **rất nhanh** (~2-3h CPU).

---

### Module C — Decomposition Loss (HIGH RISK)

**Vị trí**: `L_decomp = (CC_D)² / (1.01 + CC_B)` ở Phase I+II.

**⚠️ Risk**: Paper §12.8 đã ablate biến thể subtraction (Exp I) và chứng minh division thắng. Bất kỳ biến thể mới phải:
1. Reproduce paper baseline (form division) trước
2. Implement biến thể, train Phase I + II từ đầu (FULL retrain ~12-18h)
3. Bằng-hoặc-vượt division form

**Hypothesis**: Correlation chỉ capture linear relationship. Mutual Information / Wasserstein / IB sẽ capture relationship phức tạp hơn → decomp tốt hơn → fusion tốt hơn.

| # | Variant | Cơ chế |
|---|---|---|
| C.1 | `DecompLoss-CCDivision` | `(CC_D)²/(CC_B+ε)` (paper, baseline) |
| C.2 | `DecompLoss-CCSubtraction` | `(CC_D)² - CC_B` (paper Exp I, đã ablate) |
| C.3 | `DecompLoss-MI` | `I(D_A;D_B) - I(B_A;B_B)` |
| C.4 | `DecompLoss-InfoBottleneck` | IB principle |

**Khuyên**: chỉ làm Module C **nếu** Module A + B không đủ contribution cho ĐATN. High effort, high risk.

---

## Phase 4 — Combined model

Sau khi xong Module A và B (winners + 1 ablation matrix):

| Variant | Config | Mục đích |
|---|---|---|
| `Combined-AB` | A.winner + B.winner | Best-of-both, main contribution |
| `Combined-NoA` | A.1 (baseline sum) + B.winner | Ablation: B đóng góp bao nhiêu |
| `Combined-NoB` | A.winner + B.1 (baseline max) | Ablation: A đóng góp bao nhiêu |

Bảng ablation thesis sẽ có **8 dòng** (A.1-A.4, B.1-B.4) + 3 dòng combined → đủ cho reviewer "đẹp".

---

## Stretch goals (nếu còn thời gian)

Các hướng từ `CDDFuse_analysis.md §8` chưa thuộc Module A/B/C, **có giá trị novelty cao** nhưng effort lớn:

| ID | Tên | Lý do làm | Effort |
|---|---|---|---|
| S1 | `Loss-AnatomicalAware` (B1) | Medical-specific contribution, defense rất dễ | Medium (~6h train) |
| S2 | `Encoder-MambaHybrid` (A1) | Trending 2024-2025, novelty | Hard (~18h, full retrain) |
| S3 | `Decomp-Triband` (A2) | Architectural contribution, multi-scale | Hard |
| S4 | `Encoder-CrossModal` (A3) | Modality-aware encoder | Hard |
| S5 | `Loss-Frequency` (B3) | Quick win, ít novelty | Easy |
| S6 | `TTA inference` (C2) | +1-2% SSIM gần như free, defense yếu | Easy |

**Khuyên thứ tự nếu chọn**: **S1 (AnatLoss) > S5 (FreqLoss) > S6 (TTA) > S2/S3 (Mamba/Triband)**.

---

## Lịch trình đề xuất (10 tuần)

| Tuần | Phase | Output |
|---|---|---|
| 1 | Setup + Module A.2 (đã xong) | ✓ Gated prototype, PROGRESS.md, baseline perimage |
| 2 | Module A.3 (CrossAttn) | Variant + stats vs A.1 |
| 3 | Module A.4 (ChannelMoE) → **chốt A winner** | Friedman test 4 alternatives + CD diagram |
| 4 | Module B.2 (Mean) + B.3 (Saliency) | 2 variants light (loss-only) |
| 5 | Module B.4 (Learnable) → **chốt B winner** | Friedman test 4 alternatives |
| 6 | Combined-AB + 2 ablations | Main result của ĐATN |
| 7 | Stretch S1 (AnatLoss) | Medical contribution thứ 2 |
| 8 | Buffer + qualitative figures | Visual comparison grids |
| 9 | Viết Method + Experiments chương | Draft 1 |
| 10 | Defense prep, polish | Submit |

---

## Acceptance criteria (từ RESEARCH_METHODOLOGY.md §3)

Mỗi variant CHẤP NHẬN vào ablation table chính khi đạt **TẤT CẢ**:

- C1: Composite z ≥ baseline + 0.05
- C2: ≥ 5/22 metrics SIG sau Holm correction
- C3: Không regression > 1σ trên SSIM/QG/QY
- C4: Cliff's δ ≥ 0.33 trên ≥ 3 metrics
- C5: `_ablation_stamp.json` đầy đủ
- C6: `components_changed` mô tả rõ

Variant FAIL ≥ 2 tiêu chí → log vào `_failures/<Tag>.md`.

---

## Trạng thái hiện tại (cập nhật khi có thay đổi)

```
Phase 1 — Module A: 1/4 done (A.2 prototype, cần re-run final + 3 alternatives khác)
Phase 2 — Module B: 0/4 done
Phase 3 — Combined: pending
Stretch: pending

Bottleneck hiện tại: cần re-run baseline với --save_perimage để unlock fusion-stats workflow
```

---

## Quyết định checkpoint

Sau mỗi phase, tự hỏi:

| Checkpoint | Câu hỏi | Hành động nếu KHÔNG đạt |
|---|---|---|
| Sau Module A | Có ít nhất 1 variant SIG trên NABF? | Module A fail → focus Module B + S1 (AnatLoss) |
| Sau Module B | Có ít nhất 1 variant SIG cải thiện composite z? | Both fail → reconsider model choice (CDDFuse có thể đã sát ceiling) |
| Sau Combined | Combined > từng module đơn lẻ? | Combined không synergy → giữ winner đơn, không chấp ablation matrix |

---

## Files liên quan

- `PROGRESS.md` — log từng variant (append-only)
- `CDDFuse_analysis.md` — lý thuyết, paper §12.10 hiệu chỉnh
- `RESEARCH_METHODOLOGY.md` — quy trình + integrity rules
- `zscore_ranking.csv` — bảng so SOTA (cập nhật sau Combined)
- `_failures/` — các variant bị reject
- `_stats/` — output `fusion-stats` skill
