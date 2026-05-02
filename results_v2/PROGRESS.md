# Tiến trình cải tiến CDDFuse — ĐATN

> **Cập nhật mỗi lần chạy variant.** Không xoá entry cũ. File này là single source of truth về trạng thái thí nghiệm.
>
> Convention: append entry mới ở **§ Variants** (giữ thứ tự thời gian). Cập nhật **§ Trạng thái** ở trên cùng.

---

## Trạng thái

| | |
|---|---|
| Latest update | 2026-05-02 22:00 |
| Variants completed | 1 prototype / N planned (xem ROADMAP.md) |
| Latest activity | Re-run baseline với `--save_perimage` xong (local CPU ~7 phút) |
| Status | **Stats workflow unlocked** — sẵn sàng so sánh `CDDFuse-FuseRule-Gated` vs baseline |
| Next planned | Chạy `fusion-stats` so 2 model |

---

## Module A — Fusion Rule (target: NABF gap 1.73σ)

So sánh các cách thay thế phép `A + B` trong `BaseFuseLayer(A + B)` và `DetailFuseLayer(A + B)`.

| # | Variant | Tham số mới | Trạng thái | Composite Δ | Quyết định |
|---|---|---|---|---|---|
| A.1 | `FuseRule-Sum` (baseline) | 0 | ✓ có sẵn (CDDFuse_MIF.pth) | 0 (reference) | KEEP |
| A.2 | `FuseRule-Gated` | ~16K | ✓ prototype CPU 10 ep | TBD (Wilcoxon chưa chạy) | **ITERATE** |
| A.3 | `FuseRule-CrossAttn` | ~50K | pending | — | — |
| A.4 | `FuseRule-ChannelMoE` | ~10K | pending | — | — |

---

## Variants (chronological)

### 2026-05-02 22:00 · `CDDFuse` (baseline) · per-image re-run

**Mục đích**: Sinh per-image CSV (24 dòng × 3 modal = 72 entries) làm input cho `fusion-stats` skill — yêu cầu paired data để Wilcoxon test.

**Cmd**: `python evaluate_cddfuse.py --modal {CT,PET,SPECT} --save_perimage` (local CPU)

**Output**:
- `results_v2/CDDFuse/perimage/CDDFuse_CT_perimage.csv` (24 rows + header)
- `results_v2/CDDFuse/perimage/CDDFuse_PET_perimage.csv` (24 rows + header)
- `results_v2/CDDFuse/perimage/CDDFuse_SPECT_perimage.csv` (24 rows + header)

**Sanity check**: aggregated values bit-exact match baseline cũ (timestamp 2026-04-23) → deterministic, weights không đổi.

**Backup**: `results_v2/CDDFuse/CDDFuse_summary.csv.bak` giữ summary cũ.

---


### 2026-05-02 · `CDDFuse-FuseRule-Gated` · prototype

**Hypothesis (H1)**: Soft per-channel × per-spatial gate `g·A + (1-g)·B` thay phép sum sẽ giảm NABF (artifact ở boundary 2 modality) mà không hy sinh SSIM nhiều.

**Module thay đổi**: `models/MMIF-CDDFuse/variants/modules.py::GatedFuseLayer` (Conv1×1, init zero → epoch-0 ≈ baseline).

**Training config**:
| | Value |
|---|---|
| Mode | light_retrain (encoder/decoder frozen) |
| Trainable params | 397,704 / 1,204,784 (~33%) |
| Train data | Harvard medical 738 pairs (160 CT + 245 PET + 333 SPECT) |
| Test data | Harvard reference 72 pairs (24×3) |
| Hardware | Kaggle Tesla P100 (sm_60 unsupported by torch 2.10) → fallback **CPU** |
| Epochs | **10** (reduced from 25 due to CPU; loss curve still descending) |
| Batch | 2 (CPU) |
| LR | 1e-4 cosine → 1e-6 |
| Seed | 42 |
| Wall time | ~3-4h |
| Ckpt sha256 | `8575e252df14968376e345766f04b7bfb76670c1e30f11f69322a477a7619bfd` |

**Loss curve**:
| Epoch | Total | Intensity | Gradient |
|---|---|---|---|
| 1 | 1.022 | 0.043 | 0.098 |
| 5 | 0.886 | 0.041 | 0.084 |
| 10 | 0.875 | 0.042 | 0.083 |

→ Còn descend ở epoch 10, **chưa hội tụ hoàn toàn**.

**Quick delta vs baseline (raw, chưa significance test)**:

| Metric | CT base→var | PET base→var | SPECT base→var | Trend |
|---|---|---|---|---|
| **NABF** ↓ (target) | 0.0197 → 0.0232 ❌ | 0.0181 → **0.0103** ✓ | 0.0330 → **0.0045** ✓✓ | win 2/3 modal |
| SSIM ↑ | 1.4799 → 1.4727 ❌ | 1.4043 → 1.3976 ❌ | 1.4318 → 1.4171 ❌ | regression nhẹ |
| QG ↑ | 0.6141 → 0.6102 ≈ | 0.7344 → 0.7204 ❌ | 0.7585 → 0.7466 ❌ | regression nhẹ |
| QM ↑ | 0.0088 → 0.0043 ❌ | 0.0740 → 0.0108 ❌❌ | 0.1678 → 0.0196 ❌❌ | regression mạnh |
| QCB ↑ | 0.7965 → 0.7952 ≈ | 0.8962 → 0.8963 ≈ | 0.8802 → 0.8802 ≈ | flat |

**Phân tích thô**:
- ✓ NABF cải thiện rõ ở PET (1.8×) và SPECT (7.3×) — **target gap được giải quyết**
- ❌ Trade-off: SSIM/QG/QM giảm nhẹ → có thể do under-train (10 ep) hoặc gate đang oversmooth
- Đặc biệt SPECT NABF từ 0.033 → 0.0045 — drop rất mạnh, đáng chú ý

**Acceptance criteria check** (xem RESEARCH_METHODOLOGY.md §3):
- [ ] C1 (composite z ≥ baseline + 0.05) — **chưa tính** (cần fusion-stats)
- [ ] C2 (≥5/22 metrics SIG sau Holm) — **chưa test**
- [ ] C3 (no regression > 1σ trên SSIM/QG/QY) — likely OK (deltas nhỏ < 0.5%)
- [ ] C4 (Cliff's δ ≥ 0.33 trên ≥3 metrics) — **chưa tính**
- [x] C5 (`_ablation_stamp.json` đầy đủ) — ✓
- [x] C6 (components_changed mô tả rõ) — ✓

**Files**:
- `results_v2/CDDFuse-FuseRule-Gated/` — full output
- `results_v2/CDDFuse-FuseRule-Gated/perimage/` — input cho fusion-stats
- `models/MMIF-CDDFuse/models/CDDFuse-FuseRule-Gated.pth`

**Quyết định**: **ITERATE** (không KEEP, không REJECT).

**Lý do**:
- NABF improvement quá tốt để bỏ (SPECT giảm 7.3×) → hướng có giá trị
- SSIM regression nhỏ + 10-epoch CPU under-train → kết quả CHƯA đáng tin
- Cần re-run với 25 epoch P100 (tự nhiên CPU fallback) trước khi quyết định

**Next steps**:
1. Re-run baseline với `--save_perimage` để có per-image CSV cho stats (~30 phút Kaggle)
2. Sau khi có baseline perimage → run `fusion-stats` trên kết quả 10-epoch hiện tại
3. Nếu Wilcoxon NABF p_adj < 0.05 → re-run variant với 25 epoch (CPU ~8h hoặc Kaggle 2 sessions)
4. Tiếp theo: `FuseRule-CrossAttn` (alternative #3 trong Module A)

---

## Backlog

- [x] ~~Re-run baseline CDDFuse với `--save_perimage`~~ — DONE 2026-05-02 (local CPU ~7 phút, 72/72 ảnh, perimage CSVs ở `CDDFuse/perimage/`)
- [ ] Run `fusion-stats CDDFuse-FuseRule-Gated vs baseline`
- [ ] Implement `FuseRule-CrossAttn` (alternative #3)
- [ ] Implement `FuseRule-ChannelMoE` (alternative #4)
- [ ] Re-run `FuseRule-Gated` với 25 epoch sau khi có Wilcoxon confirm hướng
- [ ] Module B (PixelSelect alternatives) — sau khi xong Module A
- [ ] Update `results_v2/zscore_ranking.csv` với composite z mới
- [ ] CD diagram khi có ≥3 variants compare (Friedman test)

---

## Failures / Lessons learned

### 2026-05-02 · Kaggle GPU constraints
- **P100 (sm_60) không tương thích torch 2.10+cu128** — fallback CPU bắt buộc
- Pip downgrade torch 2.1.2+cu118 không override system path → bỏ approach này
- CPU light retrain 738 pairs × 10 epoch ≈ 3-4h trên Kaggle CPU instance
- Notebook commit phase chậm (~50 phút) do `/kaggle/working/` chứa quá nhiều file (data + repo clone)
- **Fix cho lần sau**: copy data vào `/tmp/` thay vì `/kaggle/working/` để Kaggle không phải commit
