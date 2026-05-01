# RESEARCH METHODOLOGY — Cải tiến CDDFuse cho Medical Image Fusion

> Kim chỉ nam thực thi cho ĐATN chuẩn NCKH. Đọc trước mỗi tuần. Không phải tài liệu kiến trúc.

---

## 1. Nguyên tắc bất biến (3 điều không bao giờ phá)

1. **Same-data ablation**. Mọi biến thể CDDFuse-X phải eval trên cùng `data/reference/` với cùng image set 24+24+24. Nếu đổi data → kết quả vô nghĩa cho ablation.
2. **Per-image metrics**. Không lưu chỉ aggregate. Không có per-image CSV ⇒ không có Wilcoxon ⇒ không có claim significance. Patch `evaluate_cddfuse.py` để có flag `--save_perimage` là việc đầu tiên.
3. **Reproducibility stamp**. Mỗi variant entry trong `results_v2/` phải có `_ablation_stamp.json` (git sha, ckpt sha, env, seed, components_changed). Nếu thiếu → entry không tồn tại với reviewer.

---

## 2. Workflow chuẩn (vòng lặp R&D)

```
┌──────────────────────────────────────────────────────────────┐
│   1. fusion-gap-analysis  →  pick 1 hướng cải tiến          │
│   2. Implement (code thay đổi trên branch riêng)             │
│   3. Train variant (log loss, save ckpt theo naming)         │
│   4. fusion-bench-variant →  results_v2/CDDFuse-<Tag>/       │
│   5. fusion-stats         →  Wilcoxon + effect size          │
│   6. Quyết định: KEEP (merge vào main contribution)          │
│                  REJECT (loại, log nguyên nhân vào failures) │
│                  ITERATE (tune hyperparam, quay lại 2)       │
│   7. Update CDDFuse_analysis.md §8 với insight mới           │
└──────────────────────────────────────────────────────────────┘
```

**Một vòng lặp đúng = 1-2 tuần**. Đừng chạy song song > 2 hướng (mất focus, ablation rối).

---

## 3. Định nghĩa "cải tiến đáng tin" (acceptance criteria)

Variant được CHẤP NHẬN vào luận văn khi đạt **TẤT CẢ**:

| # | Tiêu chí | Đo bằng |
|---|---|---|
| C1 | Composite z ≥ baseline + 0.05 | `fusion-stats` recompute zscore |
| C2 | ≥ 5/22 metrics significant sau Holm correction | `significance_*.csv`, `verdict=SIG` |
| C3 | Không regression nặng trên metric quan trọng (SSIM, QG, QY giảm > 1σ) | per-metric delta |
| C4 | Effect size ≥ Cliff's δ 0.33 (medium) trên ≥ 3 metrics | significance table |
| C5 | Có `_ablation_stamp.json` đầy đủ | check file |
| C6 | Mô tả `components_changed` rõ ràng (không "improve here and there") | manual review |

Variant FAIL ≥ 2 tiêu chí → loại, log vào `results_v2/_failures/<Tag>.md`.

---

## 4. Kế hoạch 12 tuần (chỉnh từ CDDFuse_analysis.md §9.3)

| Tuần | Mục tiêu | Output cụ thể | Skill chính |
|---|---|---|---|
| 1 | Bootstrap: patch `--save_perimage`, re-run baseline với per-image | `results_v2/CDDFuse/perimage/*` | bench-variant (cho baseline) |
| 2 | Gap analysis chính thức + chọn 3 hướng | `results_v2/_planning/gap_analysis.md` | gap-analysis |
| 3-4 | Implement + train variant #1 (e.g. AnatLoss B1 — easy quick win) | `CDDFuse-AnatLoss/` + stats | bench-variant, stats |
| 5-6 | Implement + train variant #2 (e.g. Triband A2 — main contribution) | `CDDFuse-Triband/` + stats | bench-variant, stats |
| 7-8 | Implement + train variant #3 (e.g. MambaHybrid A1 hoặc N3 LearnableSelector) | `CDDFuse-<X>/` + stats | bench-variant, stats |
| 9 | Ablation matrix: A, A+B, A+C, A+B+C | 4 variants thêm | bench-variant × 4 |
| 10 | Combined model (CDDFuse-Full) + qualitative figures | `CDDFuse-Full/` + figures | bench-variant |
| 11 | Stats consolidation, CD diagram, regression analysis | `_stats/final/` | stats |
| 12 | Viết luận văn (chương Method, Experiments, Ablation) | LaTeX draft | (next-skill) |

**Buffer**: trừ tuần 12, mọi tuần đều có thể trượt 2 ngày — đừng plan zero-slack.

---

## 5. Quản lý thí nghiệm

### Branch strategy

- `main` — code đã merge vào ablation chính.
- `variant/<tag>` — branch cho mỗi variant. Sau khi train xong: nếu KEEP → merge vào main, nếu REJECT → giữ branch + tag `failed/<tag>` để truy lại.
- Không bao giờ commit trực tiếp lên `main` khi đang train.

### Naming convention (TỔNG HỢP)

| Loại | Format | Ví dụ |
|---|---|---|
| Variant code name | `CDDFuse-<Tag>` (PascalCase) | `CDDFuse-Triband` |
| Variant branch | `variant/<tag-lower>` | `variant/triband` |
| Variant ckpt | `models/MMIF-CDDFuse/models/CDDFuse-<Tag>.pth` | |
| Variant inference script | `evaluate_cddfuse.py --variant <Tag>` | |
| Result dir | `results_v2/CDDFuse-<Tag>/` | |
| Stats run | `results_v2/_stats/<yyyymmdd_hhmm>_<tag>_vs_baseline/` | |

### Failure log format

`results_v2/_failures/<Tag>.md`:
```markdown
# CDDFuse-<Tag> — REJECTED

**Date**: 2026-XX-XX
**Hours invested**: 32h
**Hypothesis**: [1 dòng]

## Numbers
- composite_z: baseline 0.93 → variant 0.X
- SSIM: 1.439 → 1.4XX (Δ -0.0X)
- ...

## Root cause analysis
[Tại sao không work? Bài học cho hướng tiếp]

## Disposal
- Branch: variant/<tag-lower> kept, tagged failed/<tag>
- Checkpoint: deleted (giữ lại 1 sample weight nhỏ trong _failures/<Tag>_sample.pth)
```

→ Failure được ghi nhận có giá trị bằng success. Reviewer luôn hỏi "đã thử gì không work" — đây là answer.

---

## 6. Integrity rules (chống cherry-pick mà bạn không nhận ra)

1. **Không chạy lại variant 5 lần để chọn run tốt nhất**. Chạy 1 lần với seed cố định, đó là kết quả.
2. **Không filter test images "khó"**. Eval trên full set 24+24+24, không bao giờ subset trừ khi đã ablate trước (và phải log).
3. **Không tune hyperparam trên test set**. Nếu cần tune → split val/test ngay từ tuần 1 (ví dụ: 18 train per modal / 6 test). Currently project dùng 24 ảnh full as test set → sẽ phải re-design split nếu tune. **Quyết định sớm.**
4. **Không so với re-implementation tự build của model khác**. Số liệu của competitor lấy từ `results_v2/all_models_summary.csv` (đã chạy bằng cùng pipeline). KHÔNG quote số từ paper gốc của competitor — biased vì khác data split.
5. **Không silent-skip ảnh fail**. Nếu evaluate raise exception trên 1 ảnh → log rõ, KHÔNG hide → metric average không đại diện.

---

## 7. Liên kết với 3 skill

| Tình huống | Skill | Phase |
|---|---|---|
| "Tuần này làm gì?" | `fusion-gap-analysis` | Plan |
| "Variant đã train xong, đo đi" | `fusion-bench-variant` | Measure |
| "Có thật sự tốt hơn không?" | `fusion-stats` | Verify |
| "Viết section Experiments" | (skill #6 sẽ thêm sau) | Write |

3 skill này phủ Plan-Measure-Verify. Phase Write chưa có skill → vẫn manual với template trong skill #3 output.

---

## 8. Checklist trước khi nộp ĐATN

- [ ] Baseline reproducible: clone repo → 1 lệnh → ra `results_v2/CDDFuse/CDDFuse_summary.csv` matching commit baseline.
- [ ] ≥ 4 variants trong ablation table (baseline + 3 contributions + full).
- [ ] Mỗi variant có `_ablation_stamp.json` đầy đủ.
- [ ] Significance table với Holm correction trong appendix.
- [ ] CD diagram cho ranking (≥ 5 model bao gồm SOTA).
- [ ] Qualitative comparison figure (≥ 4 ảnh × ≥ 5 model).
- [ ] Failure log ≥ 1 entry (chứng tỏ đã thử và loại có khoa học).
- [ ] Code public (hoặc release-ready) — repo clean, README có 1-command repro.
- [ ] LaTeX bib entries đầy đủ cho mọi competitor đã so.
- [ ] Tự đặt 5 câu hỏi reviewer khắc nghiệt nhất + draft câu trả lời (xem CDDFuse_analysis.md §10 làm template).

---

## 9. Khi mất phương hướng

Đọc theo thứ tự:
1. `RESEARCH_METHODOLOGY.md` (file này) — quy trình.
2. `results_v2/CDDFuse_analysis.md` §8 — danh sách hướng cải tiến đã có.
3. `results_v2/zscore_ranking.csv` — vị trí hiện tại.
4. `Paper/Zhao_CDDFuse_*.pdf` — xác minh khi nghi ngờ paper claim.

Nếu vẫn mơ hồ → invoke `fusion-gap-analysis` với câu hỏi cụ thể: "tôi vừa làm xong X, gap còn lại là gì?".

---

## 10. Thước đo thành công cuối

**ĐATN chỉn chu** = reviewer trong hội đồng đọc bảng kết quả + ablation và **không hỏi**: "test này có ý nghĩa thống kê không?", "biến thể này khác baseline cụ thể ra sao?", "có cherry-pick test set không?".

3 câu hỏi đó được trả lời sẵn trong file output là đạt chuẩn NCKH. Đó là toàn bộ mục đích của methodology này.
