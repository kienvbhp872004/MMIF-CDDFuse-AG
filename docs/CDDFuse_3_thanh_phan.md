# Phân tích 3 bước cốt lõi của CDDFuse

> **Mục đích**: định vị CDDFuse theo paradigm 3 bước của image fusion truyền thống — **(1) Phân rã / (2) Tổng hợp các thành phần / (3) Biến đổi ngược** — để xác định các điểm cải tiến cho CDDFuse-AG.

---

## Sơ đồ tổng quan

```
                    ┌─ Biến đổi ảnh ─┐    ┌── Tổng hợp các thành phần ──┐    ┌─ Biến đổi ngược ─┐
                    │                │    │                              │    │                  │
   Hình ảnh         │   Phân rã      │    │  ┌─ Tổng hợp Base  ─┐        │    │                  │
   đầu vào 1   ────►│  ──────────► (f^B, f^D)  ─►  fused Base    ─►      │    │   Tái tạo lại    │   Hình ảnh
   Hình ảnh         │   Encoder      │    │                              │ ───►│    Decoder       ────► tổng hợp
   đầu vào 2   ────►│                │    │  ┌─ Tổng hợp Detail ─┐       │    │                  │
                    │                │    │     ─►  fused Detail  ─►     │    │                  │
                    └────────────────┘    └──────────────────────────────┘    └──────────────────┘
                       BƯỚC 1                     BƯỚC 2 (gồm 2 nhánh)              BƯỚC 3
```

---

## Bảng tổng hợp 3 bước

| Tiêu chí | **(1) Phân rã** | **(2) Tổng hợp thành phần** | **(3) Biến đổi ngược** |
|---|---|---|---|
| **Vai trò** | Tách ảnh $I_V, I_I$ thành cặp đặc trưng $(f^B, f^D)$ | Hợp Base + Detail của 2 modality thành cặp $(f_F^B, f_F^D)$ | Tái tạo ảnh tổng hợp $\hat{I}_F$ từ $(f_F^B, f_F^D)$ |
| **Module code** | `Restormer_Encoder` (`net.py` line 336) | `BaseFuseLayer` + `DetailFuseLayer` (`train.py`) | `Restormer_Decoder` (`net.py` line 364) |
| **Kiến trúc** | PatchEmbed + SFE (4 Transformer) + BTE + DCE (3 INN) | 2 nhánh song song: 1 Transformer (Base) + 1 INN (Detail) | Channel reduction + 4 Transformer blocks + Output head + Sigmoid |
| **Phép kết hợp** | (không áp dụng) | **$f_I + f_V$** (sum hardcoded cho cả Base & Detail) | Concat $(f_F^B, f_F^D)$ → reduce → refine |
| **Mechanism học** | Trained qua $L_{\mathrm{decomp}}$ + $L_{\mathrm{recon}}$ trong Phase I + II | Weights học qua $L_{\mathrm{fusion}}$ trong Phase II | Weights học qua $L_{\mathrm{recon}}$ trong Phase I + $L_{\mathrm{fusion}}$ trong Phase II |
| **Tham số** | ~600K (50% tổng) | Base ~50K + Detail ~30K (7% tổng) | ~500K (42% tổng) |
| **Phase huấn luyện** | Phase I + II | Phase II (epoch 40--119) | Phase I + II |
| **Loss giám sát** | $L_{\mathrm{decomp}} = \frac{(\mathrm{CC}_D)^2}{1.01 + \mathrm{CC}_B}$ | $L_{\mathrm{int}}^{II}$ (max-rule) + $L_{\mathrm{grad}}^{II}$ | $L_{\mathrm{recon}}$ (SSIM + MSE) |
| **Đặc tính kỳ vọng** | $\mathrm{CC}_B$ cao, $\mathrm{CC}_D$ thấp | Base preserve cấu trúc shared, Detail preserve texture specific | Output gần ảnh thật, smooth, no artifact |
| **Cơ chế xử lý** | Global attention (Transformer) + Local INN | Base: global attention; Detail: local INN | Channel reduce → global attention refine → sigmoid output |
| **Điểm yếu** | Không có loss explicit cho Independence trong cùng modality + Spectral property | **Phép cộng đơn giản** + max-rule loss → boundary artifact, không tận dụng tính khác biệt modality | Không có điểm yếu rõ rệt — đối xứng với Encoder |
| **Cải tiến CDDFuse-AG** | ❌ Giữ nguyên | ⭐⭐ **Adaptive Gating** (cả Base + Detail) + **Saliency-guided Pixel** (cho Detail loss) | ❌ Giữ nguyên |

---

## Chi tiết Bước 2: Tổng hợp thành phần (2 nhánh)

Bước 2 chia làm **2 nhánh song song**, một cho Base và một cho Detail. Đây là chỗ cải tiến chính của CDDFuse-AG:

| Tiêu chí | **2a. Tổng hợp Cơ sở (Base)** | **2b. Tổng hợp Chi tiết (Detail)** |
|---|---|---|
| **Đầu vào** | $f_V^B, f_I^B$ (từ Encoder) | $f_V^D, f_I^D$ (từ Encoder) |
| **Module** | `BaseFuseLayer` | `DetailFuseLayer` |
| **Kiến trúc** | 1 Transformer block (MDTA + MLP, 8 heads, dim=64) | 1 INN block (affine coupling, RealNVP-style) |
| **Phép kết hợp gốc** | **$f_I^B + f_V^B$** → Transformer refine | **$f_I^D + f_V^D$** → INN transform |
| **Tham số** | ~50K (4% tổng) | ~30K (3% tổng) |
| **Loss tương ứng** | $L_{\mathrm{int}}^{II} = \|\hat{I}_F - \max(I_V, I_I)\|^2$ | $L_{\mathrm{grad}}^{II} = \|\nabla\hat{I}_F - \max(\|\nabla I_V\|, \|\nabla I_I\|)\|_1$ |
| **Đặc tính kỳ vọng** | Base = low-frequency, shared structure | Detail = high-frequency, modality-specific texture |
| **Cơ chế xử lý** | Global attention → long-range context | Local conv + INN → giữ texture không mất info |
| **Điểm yếu** | Phép cộng coi đóng góp 2 modality như nhau | Phép cộng + max-rule tạo discontinuity ở biên |
| **Cải tiến CDDFuse-AG** | ⭐ **Adaptive Gating** thay $f_I + f_V$ | ⭐ **Adaptive Gating** + ⭐ **Saliency-guided Pixel** thay max-rule |

---

## Diễn giải chi tiết từng bước

### Bước 1: Phân rã (Decomposition)

**Vị trí**: Section 3.1 paper CDDFuse, module `Restormer_Encoder` trong `net.py`.

**Cơ chế**:
```
inp_img → PatchEmbed → SFE (4 Restormer blocks) ─┬─→ BaseFeatureExtraction  → f^B
                                                  └─→ DetailFeatureExtraction → f^D
```

Encoder **không tự phân rã** theo cách hardcoded (như wavelet truyền thống), mà học decomposition thông qua loss:

$$L_{\mathrm{decomp}} = \frac{(\mathrm{CC}_D)^2}{1.01 + \mathrm{CC}_B}$$

- **Tử số** $(\mathrm{CC}_D)^2$: phạt khi Detail của 2 modality có correlation cao → ép Detail trở thành modality-specific.
- **Mẫu số** $1.01 + \mathrm{CC}_B$: thưởng khi Base của 2 modality có correlation cao → ép Base trở thành shared structure.

**Kết luận**: CDDFuse phân rã tốt, không cần cải tiến. **CDDFuse-AG giữ nguyên** Bước 1.

---

### Bước 2: Tổng hợp thành phần (Component Fusion) — 2 nhánh

#### 2a. Tổng hợp Base
**Vị trí**: Section 3.2 paper, `BaseFuseLayer` trong `train.py`.

```python
self.BaseFuseLayer = BaseFeatureExtraction(dim=64, num_heads=8)  # 1 transformer block

# Forward Phase II:
f_F_B = BaseFuseLayer(f_I_B + f_V_B)
```

Hai bước con: phép cộng đơn giản → refine bằng Transformer block.

**Điểm yếu**: phép cộng coi đóng góp 2 modality như nhau ở mọi pixel/channel.

**CDDFuse-AG đề xuất Adaptive Gating**:
$$f_F^B = g^B \odot f_V^B + (1 - g^B) \odot f_I^B, \quad g^B = \sigma(W_g^B \cdot [f_V^B; f_I^B] + b_g^B)$$

#### 2b. Tổng hợp Detail
**Vị trí**: Section 3.2 paper, `DetailFuseLayer` trong `train.py`.

```python
self.DetailFuseLayer = DetailFeatureExtraction(num_layers=1)  # 1 INN block

# Forward Phase II:
f_F_D = DetailFuseLayer(f_I_D + f_V_D)
```

Khác Base ở 2 điểm: (1) **ít blocks hơn** (1 vs 3 trong encoder), (2) **INN thay vì Transformer**.

**Điểm yếu**: phép cộng + max-rule trong loss tạo discontinuity ở biên.

**CDDFuse-AG đề xuất 2 cải tiến**:
1. **Adaptive Gating** thay phép cộng (giống Base).
2. **Saliency-guided Pixel** thay max-rule trong loss:
$$L_{\mathrm{int}}^{II,\mathrm{AG}} = \|\hat{I}_F - (w \cdot I_V + (1-w) \cdot I_I)\|^2, \quad w = \frac{|\nabla I_V|}{|\nabla I_V| + |\nabla I_I|}$$

---

### Bước 3: Biến đổi ngược (Inverse Transform)

**Vị trí**: Section 3.1 paper, module `Restormer_Decoder` trong `net.py` line 364.

**Cơ chế**:
```python
class Restormer_Decoder(nn.Module):
    self.reduce_channel  = nn.Conv2d(128, 64, kernel_size=1)     # concat → giảm channel
    self.encoder_level2  = [TransformerBlock × 4]                # refinement
    self.output = Sequential(Conv2d(64,32,3), LeakyReLU, Conv2d(32,1,3))
    self.sigmoid = nn.Sigmoid()

# Forward:
x = concat(f_F^B, f_F^D, dim=channel)     # [B, 128, H, W]
x = reduce_channel(x)                      # [B, 64, H, W]
x = TransformerBlock × 4 (x)               # refine
x = sigmoid(output_head(x))                # [B, 1, H, W] in [0, 1]
```

Decoder gồm 3 phần:
1. **Channel Reduction**: nối $(f_F^B, f_F^D)$ rồi giảm channel từ 128 về 64 bằng Conv 1×1.
2. **Refinement**: 4 Restormer Transformer blocks (giống SFE của Encoder).
3. **Output Head**: 2 lớp Conv 3×3 (64→32→1) với LeakyReLU, kết thúc bằng sigmoid để output về $[0, 1]$.

**Đặc điểm**:
- Kiến trúc **đối xứng với Encoder** (cùng dùng Restormer blocks).
- Hai chế độ:
  - **IVF mode** (`input_img ≠ None`): có residual `output + input_img` — cho infrared-visible fusion.
  - **MIF mode** (`input_img = None`): không residual — cho medical fusion (đồ án dùng mode này).

**Loss giám sát**:
- Phase I: $L_{\mathrm{recon}}^X = 5 \cdot L_{\mathrm{SSIM}}(X, \hat{X}) + L_{\mathrm{MSE}}(X, \hat{X})$ buộc reconstruct được ảnh gốc từ $(f^B, f^D)$.
- Phase II: $L_{\mathrm{fusion}}$ buộc output gần target fused image.

**Kết luận**: Decoder đối xứng với Encoder, đảm bảo tính tái tạo. **CDDFuse-AG giữ nguyên** Bước 3.

---

## Định vị cải tiến CDDFuse-AG trên 3 bước

| Bước | Cải tiến CDDFuse-AG | Lý do |
|---|---|---|
| (1) Phân rã | ❌ Không | CDDFuse đã làm tốt với $L_{\mathrm{decomp}}$ |
| (2) Tổng hợp thành phần — Base | ⭐ Adaptive Gating thay phép cộng | Phép cộng không tận dụng tính khác biệt modality |
| (2) Tổng hợp thành phần — Detail | ⭐ Adaptive Gating + ⭐ Saliency-guided Pixel | Max-rule tạo artifact ở biên |
| (3) Biến đổi ngược | ❌ Không | Đối xứng Encoder, không có điểm yếu rõ rệt |

→ **Cải tiến tập trung 100% ở Bước 2**, đặc biệt ở 2 nhánh Base và Detail. Bước 1 và Bước 3 giữ nguyên vì đã được CDDFuse xử lý tốt.

---

## Tham chiếu code

| Bước | Class chính | File | Line |
|---|---|---|---|
| **Bước 1 — Phân rã** | | | |
| Encoder tổng | `Restormer_Encoder` | `models/MMIF-CDDFuse/net.py` | 336–362 |
| SFE (4 Transformer blocks) | `TransformerBlock × 4` | `net.py` | 306–318, 352–353 |
| BTE (Base branch) | `BaseFeatureExtraction` | `net.py` | 109–124 |
| DCE (Detail branch, 3 INN) | `DetailFeatureExtraction` | `net.py` | 167–176 |
| **Bước 2 — Tổng hợp thành phần** | | | |
| 2a. Base Fusion | `BaseFuseLayer = BaseFeatureExtraction(64, 8)` | `train.py` | line 148 init, line 222 forward |
| 2b. Detail Fusion | `DetailFuseLayer = DetailFeatureExtraction(num_layers=1)` | `train.py` | line 149 init, line 223 forward |
| **Bước 3 — Biến đổi ngược** | | | |
| Decoder tổng | `Restormer_Decoder` | `net.py` | 364–395 |
| Channel Reduction | `reduce_channel = Conv2d(128, 64, 1)` | `net.py` | 377 |
| Refinement | `encoder_level2 = [TransformerBlock × 4]` | `net.py` | 378–379 |
| Output Head | `output = Sequential(Conv 3x3, LeakyReLU, Conv 3x3) + Sigmoid` | `net.py` | 380–386 |
| **Loss** | | | |
| Decomp loss | `cc()` + `(CC_D)²/(1.01+CC_B)` | `utils/loss.py` 48 + `train.py` 133 |
| Fusion loss (max-rule) | `Fusionloss` | `utils/loss.py` | — |
