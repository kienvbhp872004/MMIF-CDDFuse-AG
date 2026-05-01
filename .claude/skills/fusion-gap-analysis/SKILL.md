---
name: fusion-gap-analysis
description: Phân tích khoảng cách (gap) giữa CDDFuse baseline và các mô hình SOTA trong results_v2/, đối chiếu với phân tích kiến trúc CDDFuse và ablation paper gốc, đề xuất hướng cải tiến có ưu tiên. Trigger khi user nói "phân tích gap", "ý tưởng cải tiến", "CDDFuse yếu ở đâu", "so sánh CDDFuse với top model", "đề xuất hướng nghiên cứu".
---

# fusion-gap-analysis

Mục đích: biến số liệu trong `results_v2/` + phân tích lý thuyết trong `results_v2/CDDFuse_analysis.md` thành **một bảng đề xuất cải tiến có thứ tự ưu tiên**, gắn metric → root-cause → giải pháp → mức effort. Đây là kim chỉ nam cho mọi quyết định "tuần này làm gì".

## Khi nào kích hoạt

Trigger:
- User nói: "phân tích gap", "CDDFuse yếu ở đâu", "ý tưởng cải tiến", "đề xuất next step", "next experiment", "đặt prioritize", "compare với MM-Net-Fusion / MFS-Fusion".
- User vừa hoàn thành 1 variant và hỏi "tiếp theo cải gì".

KHÔNG kích hoạt khi user chỉ hỏi 1 metric đơn lẻ (trả lời trực tiếp), hoặc hỏi về implementation chi tiết (đó là việc của skill khác).

## Nguồn dữ liệu (đọc theo thứ tự)

1. `results_v2/zscore_ranking.csv` — composite z-score và per-metric z (đã chuẩn hoá). Đây là nguồn rank chính.
2. `results_v2/report_overview.csv` — giá trị tuyệt đối per-model, average over modalities.
3. `results_v2/all_models_summary.csv` — per-model **per-modal** (CT/PET/SPECT). Dùng để phát hiện gap **theo modality**.
4. `results_v2/CDDFuse/CDDFuse_summary.csv` — baseline reference (3 dòng: CT/PET/SPECT).
5. `results_v2/CDDFuse_analysis.md` — phân tích kiến trúc, §7 (điểm yếu), §8 (đề xuất cải tiến A/B/C/D), §12.8 (paper ablation), §12.10 (hiệu chỉnh sau đọc paper), §12.11 (hướng N1-N4).
6. `Paper/Zhao_CDDFuse_*.pdf` — đối chiếu khi cần verify claim.

## Quy ước metric (BẮT BUỘC)

Khi tính delta/gap, áp dụng đúng chiều:

| Higher-better | Lower-better |
|---|---|
| EN, VAR, AG, SF, EI, MI_mutual, FMI, NCIE, SSIM, PSNR, QG, QM, QC, QS, QCB, QY, QMI, QSF, QNCIE, QTE | NABF, CE, RMSE, QCV |

→ **Gap chuẩn hoá** = `z_top - z_CDDFuse` đã có sẵn trong `zscore_ranking.csv` (đã đảo dấu cho lower-better). Dùng giá trị này, KHÔNG tính lại từ raw values.

## Thứ tự công việc

### Bước 1 — Snapshot vị trí hiện tại

Đọc `zscore_ranking.csv`. Trả lời 3 câu trước khi đi tiếp:
- Rank của CDDFuse là gì? (composite z)
- Top-3 models là ai? Composite z chênh CDDFuse bao nhiêu?
- Có model nào dominant trên ≥4 metrics? (cluster đối thủ chính)

Nếu CDDFuse không còn ở top-3 → flag ngay, có thể baseline đã bị stale.

### Bước 2 — Lập bảng gap per-metric

Cho từng metric trong `zscore_ranking.csv`:
- `z_CDDFuse` (cột CDDFuse)
- `z_best` (max z trong cột metric đó)
- `gap = z_best - z_CDDFuse`
- `winner` (tên model đạt z_best)

Lọc chỉ giữ các metric có `gap > 0.5σ` (đáng kể) HOẶC `z_CDDFuse < 0` (yếu rõ rệt).

### Bước 3 — Per-modality breakdown (BẮT BUỘC)

Đọc `all_models_summary.csv`. Cho mỗi metric đã flag ở Bước 2, kiểm tra:
- CDDFuse yếu **đều ở 3 modalities** hay yếu **chỉ 1 modality**?
- Nếu yếu chỉ ở SPECT → root-cause khả năng là **color/Cb-Cr handling** (xem analysis §7d, §8.D1).
- Nếu yếu chỉ ở CT → root-cause khả năng là **anatomical/structure preservation**.

→ Modal-specific gaps có giá trị novelty cao hơn gap chung — ưu tiên đề cập.

### Bước 4 — Map gap → root cause → fix

Đối chiếu gap với phân tích trong `CDDFuse_analysis.md`. Dùng bảng map sau:

| Metric yếu | Root cause khả nghi (theo §7) | Fix candidate (theo §8/§12.11) |
|---|---|---|
| SSIM, PSNR, RMSE thấp | (c) Fusion rule ngây thơ A+B; (a) no multi-scale | C1 (learnable fusion gate), A2 (3-band), N3 (learnable selector thay max) |
| QG, EI, AG thấp (gradient/edge yếu) | (b) Decomposition cứng 2-band; (a) no multi-scale | A2 (3-band decomp), B3 (frequency-domain loss) |
| QCB, QCV xấu (color/contrast) | (d) Color không xử lý native; (e) encoder shared | D1 (joint Y-CbCr), A3 (modality-aware encoder) |
| MI_mutual, FMI, QMI thấp (info preservation) | (f) Loss đơn giản; INN dim hạn chế | N1 (MI-based decomp loss), B1 (anatomical-aware loss) |
| NABF cao (artifact) | (c) Fusion rule + (a) single-scale | C1 (gated fusion), C2 (TTA), A2 |
| CE cao, QTE thấp (entropy/info-theoretic) | Loss không có info-theoretic term | N1 (MI-based / IB-based loss) |

### Bước 5 — Áp dụng filter "paper-aware"

Trước khi đề xuất, **CHECK paper ablation §12.8**:

1. **L_decomp form**: Paper đã chứng minh `(CC_D)²/(CC_B+ε)` > subtraction. → Nếu định đề xuất sửa L_decomp, phải so với form division, KHÔNG so với form rỗng.
2. **LT trong BTE, INN trong DCE**: Paper ablation cho thấy mỗi vai trò là tối ưu. → Đề xuất swap (ví dụ: thay LT bằng Mamba) phải có **separate ablation** cho BTE và DCE — KHÔNG được swap cả hai cùng lúc rồi report 1 con số.
3. **Two-stage training**: Paper chứng minh critical (EN -6.3% khi remove). → KHÔNG đề xuất "merge thành 1-stage" mà không có thiết kế bù lại.
4. **Shared SFE**: Paper biện luận self-attention across feature dim đã đủ cross-modality. → Đề xuất 2-encoder phải defense lý do.

→ Skill này **TỪ CHỐI đưa ra đề xuất** vi phạm các điểm trên trừ khi user yêu cầu rõ "tôi muốn challenge paper ở điểm này".

### Bước 6 — Score đề xuất

Cho mỗi candidate, tính 3 trục:

| Trục | Giá trị 1 | Giá trị 2 | Giá trị 3 |
|---|---|---|---|
| **Effort** (giờ implement + train) | < 8h (Easy) | 1-3 ngày (Medium) | 1-2 tuần (Hard) |
| **Expected gain** (∆composite_z dự kiến) | < 0.05 | 0.05–0.15 | > 0.15 |
| **Novelty** (review-worthy) | Engineering trick | Combine known ideas | Có theoretical contribution |

Output: bảng N candidates × 3 trục, sort theo `gain × novelty / effort`.

### Bước 7 — Sanity check trước khi báo cáo

- Có ít nhất 1 đề xuất "quick-win" (Effort=Easy, gain ≥ 0.05) để user làm tuần đầu? Nếu không → giảm threshold.
- Có ít nhất 1 đề xuất "main contribution" (Novelty=High) để làm xương sống luận văn? Nếu không → flag.
- Tổng 3-5 đề xuất, KHÔNG dump 10 cái — quá nhiều = vô dụng.

## Format output BẮT BUỘC

```markdown
## Vị trí hiện tại
- CDDFuse rank #X / 23, composite z = +Y.YY
- Top-3: [list, kèm gap z]
- Modalities yếu nhất: [CT/PET/SPECT]

## Top metric gaps
| Metric | z_CDDFuse | z_top | Winner | Gap | Modal yếu nhất |
|---|---|---|---|---|---|
...

## Đề xuất ưu tiên (3-5 candidates)

### #1 [Tên, ví dụ "Learnable fusion gate (C1)"]
- **Target gap**: SSIM (-0.X), QG (-0.Y) ở modal Z
- **Root cause hypothesis**: [1 dòng]
- **Cơ chế fix**: [2-3 dòng]
- **Effort/Gain/Novelty**: M / 0.10 / Medium
- **Reference**: §8.C1 trong analysis, paper §12.8 row III
- **Risk**: [paper ablation đã thử biến thể nào? cần defense gì?]
- **Acceptance criterion**: composite_z ↑ ≥ 0.05 và Wilcoxon p < 0.05 vs baseline

### #2 ...

## Cảnh báo / Caveat
- [Bất kỳ điểm nào skill phát hiện baseline đã stale, hoặc đề xuất xung đột với paper ablation]
```

## Hard rules

- KHÔNG đề xuất hướng KHÔNG có trong `CDDFuse_analysis.md` §8 hoặc §12.11, trừ khi giải thích được lý do mới (insight từ data).
- KHÔNG đề xuất "thử cả 5 hướng cùng lúc". Chọn 3-5, có rationale.
- KHÔNG quote z-score đến 4 chữ số thập phân — chỉ 2 chữ số (giả độ chính xác).
- KHÔNG bỏ qua per-modality view — gap CT vs gap PET vs gap SPECT khác nhau hoàn toàn về root cause.
- LUÔN gắn mỗi đề xuất với 1 acceptance criterion đo được.

## Hậu quả nếu làm sai

Nếu skill này đưa ra đề xuất không bám paper / không bám data → user mất 2-4 tuần làm hướng sai → hỏng tiến độ ĐATN. Ưu tiên thẳng thắn ("không có gap đáng kể trên metric X — bỏ qua") hơn là dàn trải.
