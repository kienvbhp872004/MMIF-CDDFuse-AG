---
name: fusion-bench-variant
description: Đánh giá một CDDFuse variant (cải tiến) trên 3 modality MRI-CT/PET/SPECT theo đúng quy trình reproducible — sinh per-image metrics, summary CSV, ablation stamp, và so sánh delta vs baseline. Trigger khi user nói "đánh giá variant", "chạy CDDFuse-X", "benchmark variant mới", "test cải tiến".
---
# fusion-bench-variant

Mục đích: chuyển 1 thay đổi code/loss/training của user thành **một entry chuẩn trong `results_v2/`** đủ để (a) so sánh thống kê (skill `fusion-stats` cần per-image), (b) đặt vào bảng ablation luận văn, (c) reviewer có thể truy ngược chính xác cấu hình thí nghiệm.

## Khi nào kích hoạt

Trigger:

- "đánh giá CDDFuse-Mamba" / "benchmark variant X" / "chạy variant <tên>" / "test cải tiến Y" / "thêm vào kết quả".
- Sau khi user vừa modify `models/MMIF-CDDFuse/net.py` hoặc `utils/loss.py` và muốn đo.

KHÔNG kích hoạt cho:

- Re-run baseline thuần (CDDFuse gốc) — đã có `results_v2/CDDFuse/`.
- Quick smoke test (< 5 ảnh) — dùng `--max_images` thẳng.

## Naming convention BẮT BUỘC

Variant name format: **`CDDFuse-<Tag>`** với `<Tag>` là PascalCase mô tả ngắn.

Ví dụ hợp lệ:

- `CDDFuse-MambaHybrid` (A1)
- `CDDFuse-Triband` (A2)
- `CDDFuse-AnatLoss` (B1)
- `CDDFuse-FreqLoss` (B3)
- `CDDFuse-GatedFuse` (C1)
- `CDDFuse-Full` (kết hợp A1+A2+B1)
- `CDDFuse-AblA` (ablation variant A)

Không hợp lệ: `CDDFuse_v2`, `MyCDDFuse`, `CDDFuse-test1` (không truy được scope).

## Pre-flight checklist (CHẠY TRƯỚC KHI BENCHMARK)

Skill PHẢI verify đủ 7 điểm trước khi cho phép chạy. Thiếu điểm nào → BLOCK và yêu cầu user bổ sung.

1. **Variant name** đúng format trên.
2. **Checkpoint variant** tồn tại tại `models/MMIF-CDDFuse/models/CDDFuse-<Tag>.pth` (hoặc đường dẫn user chỉ định, phải absolute).
3. **Code thay đổi đã được commit** (`git status` clean trên `models/MMIF-CDDFuse/`). Nếu dirty → ép user commit hoặc stash trước. Reproducibility KHÔNG đàm phán.
4. **Inference script tồn tại**: hoặc `models/MMIF-CDDFuse/evaluate_cddfuse.py` đã có flag `--variant <Tag>` (nếu modular), hoặc `models/MMIF-CDDFuse/evaluate_cddfuse_<tag>.py` (nếu khác kiến trúc nhiều).
5. **Output dir chưa tồn tại** tại `results_v2/CDDFuse-<Tag>/`. Nếu tồn tại → hỏi user `--force` (xoá) hay `--rerun` (giữ Fusion images, tính lại metric) hay abort.
6. **Same data**: confirm dùng `data/reference/` (giống baseline). KHÔNG dùng dataset khác — ablation phải same-pairs same-images.
7. **Seed cố định**: variant script phải set `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`. Nếu chưa → patch trước khi chạy.

Pre-flight output (in trước khi chạy):

```
[bench-variant] Variant:        CDDFuse-<Tag>
[bench-variant] Checkpoint:     <path> (size: X MB, mtime: Y)
[bench-variant] Git commit:     <short_sha> (clean=True)
[bench-variant] Data root:      data/reference/
[bench-variant] Out dir:        results_v2/CDDFuse-<Tag>/  (new)
[bench-variant] Modalities:     CT, PET, SPECT (24+24+24 = 72 pairs)
[bench-variant] Seed:           42 (torch, numpy, random)
[bench-variant] Device:         cuda / cpu
```

## Pipeline thực thi

### Bước 1 — Chuẩn bị output dir

```
results_v2/CDDFuse-<Tag>/
├── Fusion/                          # ảnh fused per-modal subfolder hoặc flat
│   ├── CT/
│   ├── PET/
│   └── SPECT/
├── perimage/                        # PER-IMAGE metrics (REQUIRED)
│   ├── CDDFuse-<Tag>_CT_perimage.csv
│   ├── CDDFuse-<Tag>_PET_perimage.csv
│   └── CDDFuse-<Tag>_SPECT_perimage.csv
├── CDDFuse-<Tag>_summary.csv        # average per-modal (3 rows)
├── CDDFuse-<Tag>_<modal>_summary_<ts>.json   # per-modal JSON
├── _ablation_stamp.json             # config diff vs baseline (CRITICAL)
└── _runlog.txt                      # stdout/stderr full trace
```

### Bước 2 — Chạy 3 modality

Dùng `run_all_v2.py` style hoặc subprocess thẳng. Mỗi modal:

```
python models/MMIF-CDDFuse/evaluate_cddfuse.py \
    --variant <Tag> \
    --modal {CT|PET|SPECT} \
    --harvard_root data/reference/ \
    --out_dir results_v2/CDDFuse-<Tag>/ \
    --ckpt models/MMIF-CDDFuse/models/CDDFuse-<Tag>.pth \
    --save_perimage  # FLAG MỚI cần thêm — xem bước 3
```

Nếu evaluate script chưa hỗ trợ `--save_perimage` → patch trước (chỉ cần dump `all_rows` ra `perimage/<modal>_perimage.csv` với cột `image,EN,VAR,...`). KHÔNG bỏ qua bước này — skill `fusion-stats` phụ thuộc.

### Bước 3 — Đảm bảo per-image CSV

File `perimage/CDDFuse-<Tag>_<modal>_perimage.csv` **BẮT BUỘC** có:

```
image,EN,MI,VAR,AG,SF,EI,NCIE,MI_mutual,NABF,FMI,CE,SSIM,PSNR,RMSE,QG,QM,QC,QS,QCB,QCV,QY,QMI,QSF,QNCIE,QTE
```

Mỗi dòng = 1 ảnh. Đây là input chuẩn cho Wilcoxon test (skill #3).

Nếu thiếu file này → output không hợp lệ, BLOCK ghi vào summary.

### Bước 4 — Sinh ablation stamp

File `results_v2/CDDFuse-<Tag>/_ablation_stamp.json`:

```json
{
  "variant_name": "CDDFuse-<Tag>",
  "based_on": "CDDFuse",
  "git_commit": "<short_sha>",
  "git_clean": true,
  "datetime": "2026-04-30T...",
  "components_changed": [
    {"file": "models/MMIF-CDDFuse/net.py",
     "module": "BaseFeatureExtraction",
     "change": "replaced with HybridMambaBlock (mamba-ssm 2.x)"}
  ],
  "hyperparams_changed": {
    "lr": [1e-4, 1e-4],
    "batch_size": [8, 8],
    "epochs_phase1": [40, 40],
    "epochs_phase2": [80, 80],
    "alpha_decomp": [2.0, 2.0]
  },
  "checkpoint": {
    "path": "models/MMIF-CDDFuse/models/CDDFuse-<Tag>.pth",
    "sha256": "<hash>",
    "size_mb": 12.3,
    "training_log": "<path or null>"
  },
  "env": {
    "python": "3.X.Y",
    "torch": "2.x.y",
    "cuda": "12.x",
    "gpu": "RTX 3090" 
  },
  "seed": 42,
  "data": {"root": "data/reference/", "n_pairs": {"CT": 24, "PET": 24, "SPECT": 24}}
}
```

User phải fill **`components_changed`** thủ công (skill nhắc, KHÔNG tự suy diễn từ git diff — quá rủi ro mô tả sai).

### Bước 5 — Update central CSVs

Sau khi 3 modal xong, append rows vào:

- `results_v2/all_models_summary.csv` (3 dòng mới: model=`CDDFuse-<Tag>`, modal in {CT,PET,SPECT})
- KHÔNG đụng `report_overview.csv` và `zscore_ranking.csv` — recompute đó là việc của tools riêng (skill #3).

### Bước 6 — Delta report (in ra stdout + lưu)

So sánh `CDDFuse-<Tag>` vs `CDDFuse` baseline:

```
[bench-variant] === Delta vs CDDFuse baseline ===
                  CT          PET         SPECT       AVG
SSIM    base   1.4799      1.4043      1.4318      1.4387
        new    1.4XXX      1.4XXX      1.4XXX      1.4XXX
        Δ      +0.00XX     +0.00XX     +0.00XX     +0.00XX
QG      ...
...
[bench-variant] Composite Δ (across 22 metrics): +0.0X (sign of mean z-delta)
[bench-variant] CAVEAT: significance NOT yet tested. Run skill `fusion-stats` next.
```

Lưu ra `results_v2/CDDFuse-<Tag>/_delta_report.md`.

### Bước 7 — Append vào PROGRESS.md (BẮT BUỘC)

Sau khi 3 modal xong + có delta report, **append entry mới** vào `results_v2/PROGRESS.md` theo template:

```markdown
### YYYY-MM-DD · `CDDFuse-<Tag>` · <prototype|final>

**Hypothesis**: ...

**Module thay đổi**: file::class

**Training config**: bảng (mode, params, data, hardware, epochs, batch, lr, seed, wall time, ckpt sha)

**Loss curve**: bảng (epoch 1, mid, last) — đọc từ `<variant>_train_history.json`

**Quick delta vs baseline**: bảng metrics × modals từ `_delta_report.md`

**Acceptance criteria check** (xem RESEARCH_METHODOLOGY.md §3): checklist

**Files**: list paths

**Quyết định**: KEEP / ITERATE / REJECT + lý do

**Next steps**: ...
```

Cũng update `## Trạng thái` block ở đầu PROGRESS.md (latest update, latest variant, status).

KHÔNG xoá entry cũ. KHÔNG để rỗng acceptance criteria — nếu chưa test, ghi rõ "chưa tính".

### Bước 8 — Hand-off

Cuối cùng in:

```
[bench-variant] DONE. Next steps:
  1. Run `fusion-stats` for significance tests vs baseline
  2. Update `results_v2/zscore_ranking.csv` if delta meaningful
  3. Review `results_v2/PROGRESS.md` (auto-updated)
  4. Add row to thesis ablation table (Section X.Y)
```

## Hard rules

- KHÔNG ghi đè `results_v2/CDDFuse/` (baseline) trong bất kỳ trường hợp nào.
- KHÔNG bỏ qua per-image CSV — đó là dữ liệu duy nhất cho test thống kê paired.
- KHÔNG chạy với git dirty trên `models/MMIF-CDDFuse/`. Reproducibility là điều kiện cần của ĐATN chỉn chu.
- KHÔNG dùng dataset/split khác baseline. Ablation = same-data, only-component-changes.
- KHÔNG self-fill `components_changed` — yêu cầu user xác nhận. Sai mô tả là sai paper.
- LUÔN sinh `_ablation_stamp.json` — đây là tài liệu đính kèm khi reviewer hỏi "variant này thực sự là gì".
- LUÔN append vào `results_v2/PROGRESS.md` ngay khi variant xong (kể cả khi prototype). User cần single source of truth để track tiến độ ĐATN.
- LUÔN check checkpoint sha256 — nếu trùng với baseline checkpoint, BLOCK ngay (nghĩa là user quên replace ckpt).

## Khi user hỏi "tại sao phức tạp vậy"

Trả lời ngắn:

> ĐATN chỉn chu ≈ paper đầu tay. Reviewer sẽ hỏi: "model con của bạn khác gốc cụ thể ra sao, train với seed nào, ablation có same-data không, p-value thế nào". Nếu thiếu 1 điểm → defense lúng túng. 7 phút setup pre-flight đổi 7 ngày tránh phải re-run.

## Output cuối cùng (preview)

Sau khi skill chạy xong, user có:

1. `results_v2/CDDFuse-<Tag>/` đầy đủ ảnh + per-image CSV + summary + stamp
2. 3 dòng mới trong `all_models_summary.csv`
3. Delta report so với baseline
4. Sẵn sàng cho `fusion-stats`
