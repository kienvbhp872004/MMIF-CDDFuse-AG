# Phân tích 3 thành phần cốt lõi của CDDFuse

> **Mục đích**: xác định rõ ràng 3 thành phần trong CDDFuse theo paradigm 3 bước (phân rã / tổng hợp cơ sở / tổng hợp chi tiết) để định vị các điểm cải tiến cho CDDFuse-AG.

---

## Bảng tổng hợp so sánh 3 thành phần

| Tiêu chí | **(1) Phân rã** | **(2) Tổng hợp Cơ sở (Base)** | **(3) Tổng hợp Chi tiết (Detail)** |
|---|---|---|---|
| **Vai trò** | Tách ảnh đầu vào $I_V, I_I$ thành cặp đặc trưng $(f^B, f^D)$ | Hợp Base của 2 modality thành $f_F^B$ | Hợp Detail của 2 modality thành $f_F^D$ |
| **Module trong code** | `Restormer_Encoder` (`net.py` line 336–362) | `BaseFuseLayer` (`train.py` line 148) | `DetailFuseLayer` (`train.py` line 149) |
| **Kiến trúc cụ thể** | PatchEmbed + SFE (4 Restormer blocks) + BTE (1 Transformer) + DCE (3 INN blocks) | 1 Transformer block (MDTA + MLP), 8 heads, dim=64 | 1 INN block (affine coupling, RealNVP-style) |
| **Phép kết hợp gốc** | (không áp dụng — output từ Encoder) | **$f_I^B + f_V^B$ → Transformer refine** (sum hardcoded) | **$f_I^D + f_V^D$ → INN transform** (sum hardcoded) |
| **Mechanism học** | Trained via $L_{\mathrm{decomp}}$ + $L_{\mathrm{recon}}$ trong Phase I + II | Transformer block weights học qua $L_{\mathrm{fusion}}$ trong Phase II | INN block weights học qua $L_{\mathrm{fusion}}$ trong Phase II |
| **Tham số học** | ~600K (50% tổng) | ~50K (4% tổng) | ~30K (3% tổng) |
| **Phase huấn luyện** | Phase I (epoch 0–39) + tiếp tục fine-tune Phase II | Phase II (epoch 40–119) | Phase II (epoch 40–119) |
| **Loss giám sát chính** | $L_{\mathrm{decomp}} = \frac{(\mathrm{CC}_D)^2}{1.01 + \mathrm{CC}_B}$ + $L_{\mathrm{recon}}$ + $L_{\mathrm{TV}}$ | $L_{\mathrm{int}}^{II} = \|\hat{I}_F - \max(I_V, I_I)\|^2$ (max-rule) | $L_{\mathrm{grad}}^{II} = \|\nabla \hat{I}_F - \max(\|\nabla I_V\|, \|\nabla I_I\|)\|_1$ |
| **Đặc tính kỳ vọng** | $\mathrm{CC}_B$ cao (Base shared), $\mathrm{CC}_D$ thấp (Detail specific) | Base = low-frequency, shared structure | Detail = high-frequency, modality-specific texture |
| **Cơ chế xử lý** | Global attention (Transformer) + Local conv (INN) | Global attention → long-range context cho cấu trúc tổng thể | Local conv + INN → giữ texture không mất info |
| **Điểm yếu** | Không có loss explicit cho Independence trong cùng modality + Spectral property | **Phép cộng đơn giản** → không tận dụng được tính khác biệt của modality | **Phép cộng đơn giản** + max-rule loss tạo discontinuity ở biên |
| **Cải tiến CDDFuse-AG đề xuất** | ❌ Giữ nguyên (CDDFuse đã làm tốt) | ⭐ **Adaptive Gating** thay phép cộng | ⭐ **Adaptive Gating** thay phép cộng + **Saliency-guided Pixel** thay max-rule |

---

## Diễn giải chi tiết từng thành phần

### Thành phần 1: Phân rã (Decomposition)

**Vị trí**: Section 3.1 của paper CDDFuse, module `Restormer_Encoder` trong `net.py`.

**Cơ chế**:
```
inp_img → PatchEmbed → SFE (4 Restormer blocks) ─┬─→ BaseFeatureExtraction  → f^B
                                                  └─→ DetailFeatureExtraction → f^D
```

Encoder **không tự phân rã** theo cách hardcoded (như wavelet truyền thống), mà học decomposition thông qua loss:

$$L_{\mathrm{decomp}} = \frac{(\mathrm{CC}_D)^2}{1.01 + \mathrm{CC}_B}$$

- **Tử số** $(\mathrm{CC}_D)^2$: phạt khi Detail của 2 modality có correlation cao → ép Detail trở thành modality-specific.
- **Mẫu số** $1.01 + \mathrm{CC}_B$: thưởng khi Base của 2 modality có correlation cao → ép Base trở thành shared structure.

**Kết luận**: CDDFuse phân rã tốt, chưa cần cải tiến. **CDDFuse-AG giữ nguyên** thành phần này.

---

### Thành phần 2: Tổng hợp Cơ sở (Base Fusion)

**Vị trí**: Section 3.2 của paper, `BaseFuseLayer` trong `train.py`.

**Cơ chế**:
```python
self.BaseFuseLayer = BaseFeatureExtraction(dim=64, num_heads=8)  # 1 transformer block

# Forward Phase II:
f_F_B = BaseFuseLayer(f_I_B + f_V_B)
```

Hai bước:
1. **Phép cộng đơn giản** `f_I^B + f_V^B` (heuristic, không học)
2. Refine bằng 1 transformer block (MDTA + Mlp)

**Đặc điểm**:
- Tham số học **chỉ ở transformer block**, không có tham số học cho phép kết hợp.
- Dùng **global attention** vì Base = low-frequency, cần long-range context.

**Điểm yếu**: Phép cộng coi đóng góp 2 modality như nhau ở mọi pixel, mọi channel — không tận dụng được tính chất khác biệt giữa chúng.

**CDDFuse-AG đề xuất**: thay phép cộng bằng **Adaptive Gating**:
$$f_F^B = g^B \odot f_V^B + (1 - g^B) \odot f_I^B, \quad g^B = \sigma(W_g^B \cdot [f_V^B; f_I^B] + b_g^B)$$

---

### Thành phần 3: Tổng hợp Chi tiết (Detail Fusion)

**Vị trí**: Section 3.2 của paper, `DetailFuseLayer` trong `train.py`.

**Cơ chế**:
```python
self.DetailFuseLayer = DetailFeatureExtraction(num_layers=1)  # 1 INN block (vs 3 trong encoder)

# Forward Phase II:
f_F_D = DetailFuseLayer(f_I_D + f_V_D)
```

Hai bước:
1. **Phép cộng đơn giản** `f_I^D + f_V^D` (heuristic, giống Base)
2. Đưa qua 1 INN block (affine coupling)

**Đặc điểm**:
- Dùng **INN (invertible)** vì high-freq detail nhạy với info loss — INN preserves information.
- Áp dụng **local convolution** (1×1 + 3×3 depthwise) vì Detail = local texture.
- Khác Base ở 2 điểm: (1) **ít blocks hơn** (1 vs 3), (2) **INN thay vì Transformer**.

**Điểm yếu**:
1. Phép cộng `f_I^D + f_V^D` đơn giản, giống Base.
2. **Quy tắc max-pixel** trong loss $L_{\mathrm{int}}^{II} = \|\hat{I}_F - \max(I_V, I_I)\|^2$ tạo discontinuity ở biên → boundary artifact.

**CDDFuse-AG đề xuất 2 cải tiến**:
1. **Adaptive Gating** thay phép cộng (giống thành phần 2).
2. **Saliency-guided Pixel** thay max-rule:
$$L_{\mathrm{int}}^{II,\mathrm{AG}} = \|\hat{I}_F - (w \cdot I_V + (1-w) \cdot I_I)\|^2, \quad w = \frac{|\nabla I_V|}{|\nabla I_V| + |\nabla I_I|}$$

---

## Định vị cải tiến CDDFuse-AG trên 3 thành phần

| Thành phần | Cải tiến của CDDFuse-AG | Mức độ thay đổi |
|---|---|---|
| (1) Phân rã | ❌ Không | Giữ 100% như CDDFuse gốc |
| (2) Tổng hợp Base | ⭐ Adaptive Gating thay phép cộng | Architecture-level, thêm ~16K params |
| (3) Tổng hợp Detail | ⭐ Adaptive Gating + ⭐ Saliency-guided Pixel | Architecture + Loss-level |

→ Cải tiến **tập trung 100% ở Bước 2** (tổng hợp), tách rõ Base path vs Detail path.

---

## Ghi chú tham khảo code

| Thành phần | Class chính | File | Line |
|---|---|---|---|
| Phân rã | `Restormer_Encoder` | `models/MMIF-CDDFuse/net.py` | 336–362 |
| Phân rã — SFE | `TransformerBlock × 4` | `net.py` | 306–318, 352–353 |
| Phân rã — BTE | `BaseFeatureExtraction` | `net.py` | 109–124 |
| Phân rã — DCE | `DetailFeatureExtraction` (3 INN) | `net.py` | 167–176 |
| Tổng hợp Base | `BaseFuseLayer = BaseFeatureExtraction(64, 8)` | `train.py` | line 148 (init), line 222 (forward) |
| Tổng hợp Detail | `DetailFuseLayer = DetailFeatureExtraction(num_layers=1)` | `train.py` | line 149 (init), line 223 (forward) |
| Loss decomp | `cc()` + công thức `(CC_D)²/(1.01+CC_B)` | `utils/loss.py` line 48 + `train.py` line 133 |
| Loss intensity (max-rule) | `Fusionloss` | `utils/loss.py` | — |
