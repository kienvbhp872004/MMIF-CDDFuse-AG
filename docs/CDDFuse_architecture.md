# CDDFuse — Kiến trúc chi tiết

**Paper**: Zhao et al., *CDDFuse: Correlation-Driven Dual-Branch Feature Decomposition for Multi-Modality Image Fusion*, **CVPR 2023**.

**Source code**: `models/MMIF-CDDFuse/net.py`

---

## 1. Tổng quan

CDDFuse là kiến trúc **dual-branch transformer-CNN** cho tổng hợp ảnh đa phương thức (multi-modality image fusion). Ý tưởng cốt lõi: **decompose** mỗi ảnh thành **Base feature** (shared low-frequency, global semantics) và **Detail feature** (modality-specific high-frequency, local texture), rồi fuse riêng từng nhánh.

### Dòng chảy chính

```
Input ảnh I_V (visible/source)  ─┐
Input ảnh I_I (infrared/MRI)    ─┤
                                  │
              ┌───────────────────┼─────────────────────┐
              │                   │                     │
              ▼                   ▼                     │
        ┌──────────┐        ┌──────────┐                │
        │ Encoder  │        │ Encoder  │                │  (shared weights)
        │ (chung)  │        │ (chung)  │                │
        └────┬─────┘        └────┬─────┘                │
             │                   │                       │
   ┌─────────┼──────────┐ ┌──────┼─────────┐            │
   ▼         ▼          ▼ ▼      ▼         ▼            │
  f_V_B    f_V_D     f_V f_I   f_I_B    f_I_D           │
 (Base)  (Detail)        │     (Base)   (Detail)        │
                                                         │
        ┌────────┐    ┌────────────┐                    │
        │ FuseB  │    │   FuseD    │                    │
        │ (Base) │    │ (Detail)   │                    │
        └────┬───┘    └─────┬──────┘                    │
             ▼              ▼                            │
            f_F_B         f_F_D                          │
              ╲             ╱                            │
               ╲           ╱                             │
                ▼         ▼                              │
              ┌─────────────┐                            │
              │   Decoder   │                            │
              └──────┬──────┘                            │
                     ▼                                   │
                 Fused image                             │
```

---

## 2. Encoder — `Restormer_Encoder`

**Vai trò**: extract shared shallow features rồi decompose thành 2 nhánh Base/Detail.

### Cấu trúc

```python
class Restormer_Encoder(nn.Module):
    inp_channels=1, out_channels=1, dim=64
    num_blocks=[4, 4], heads=[8, 8, 8]
    ffn_expansion_factor=2

    self.patch_embed     = OverlapPatchEmbed(1, 64)
    self.encoder_level1  = [TransformerBlock(dim=64, heads=8) × 4]    # SFE
    self.baseFeature     = BaseFeatureExtraction(dim=64, heads=8)     # BTE branch
    self.detailFeature   = DetailFeatureExtraction(num_layers=3)      # DCE branch
```

### Forward

```python
x = PatchEmbed(input)                          # [B, 1, H, W] -> [B, 64, H, W]
shallow = TransformerBlock × 4 (x)             # SFE: cross-modality shared features
f_B = BaseFeatureExtraction(shallow)           # global low-freq
f_D = DetailFeatureExtraction(shallow)         # local high-freq
return f_B, f_D, shallow
```

### Thành phần cụ thể

#### 2.1 OverlapPatchEmbed
- `Conv2d(1, 64, kernel=3, stride=1, padding=1)` — biến ảnh 1 kênh xám thành 64 kênh feature giữ nguyên kích thước H×W.

#### 2.2 SFE — Shallow Feature Extraction (`encoder_level1`)
- **4 Restormer TransformerBlock** với 8 attention heads.
- Mỗi block:
  ```
  x = x + Attention(LayerNorm(x))     # MDTA (Multi-Dconv Transposed Attention)
  x = x + FeedForward(LayerNorm(x))   # GDFN (Gated-Dconv Feed Network)
  ```
- **MDTA** = `Attention` class trong code: tính attention theo **channel** (không theo spatial) — O(C²) thay O(N²) — efficient cho ảnh độ phân giải cao.
- **GDFN** = `FeedForward` class: depthwise conv 3×3 + gating (`GELU(x1)*x2`).

#### 2.3 BTE — Base Transformer Encoder (`BaseFeatureExtraction`)
**Một** TransformerBlock-like, dùng `AttentionBase` (vanilla multi-head spatial attention) thay MDTA:
```python
class BaseFeatureExtraction(nn.Module):
    self.attn = AttentionBase(dim=64, num_heads=8)        # spatial MHA
    self.mlp  = Mlp(dim=64, ffn_expansion_factor=1.0)     # FFN

def forward(x):
    x = x + self.attn(LayerNorm(x))
    x = x + self.mlp(LayerNorm(x))
```
- **Vai trò**: học **long-range global context** (low-frequency, shared structure giữa 2 modality).
- Output: `f_B` [B, 64, H, W].

#### 2.4 DCE — Detail CNN Encoder (`DetailFeatureExtraction`)
**3 Invertible Neural Network (INN) blocks** xếp tuần tự:
```python
class DetailFeatureExtraction(nn.Module):
    self.net = nn.Sequential(*[DetailNode() × 3])

def forward(x):  # x: [B, 64, H, W]
    z1, z2 = split_along_channel(x)              # [B, 32, H, W] mỗi
    for layer in DetailNodes:
        z1, z2 = layer(z1, z2)
    return concat(z1, z2)                        # [B, 64, H, W]
```

Mỗi `DetailNode` là **affine coupling layer** (RealNVP-style INN):
```python
z1, z2 = split(shuffleconv(concat(z1, z2)))      # channel shuffle
z2 = z2 + θ_φ(z1)                                # additive coupling
z1 = z1 * exp(θ_ρ(z2)) + θ_η(z2)                 # affine coupling
return z1, z2
```
- `θ_φ`, `θ_ρ`, `θ_η` = `InvertedResidualBlock` (MobileNet-style 1×1 → 3×3 dw → 1×1).
- **Vai trò INN**: invertible mapping → giữ nguyên thông tin (no info loss), phù hợp **high-frequency detail** vốn nhạy với compression.

---

## 3. Fusion layers

### `BaseFuseLayer` = `BaseFeatureExtraction(dim=64, heads=8)`
Một TransformerBlock-like xử lý feature đã merge.

### `DetailFuseLayer` = `DetailFeatureExtraction(num_layers=1)`
INN block (chỉ 1 layer, khác Encoder có 3).

### Phase II forward (từ `train.py`)
```python
f_F_B = BaseFuseLayer(f_I_B + f_V_B)             # ← phép cộng đơn giản
f_F_D = DetailFuseLayer(f_I_D + f_V_D)
```

**⚠️ Điểm quan trọng**: paper thực hiện fuse bằng **phép cộng feature** `f_I + f_V` rồi đưa qua transformer/INN. Không có gating, attention, hoặc selection — đây chính là **điểm yếu** mà Module A (Gated, CrossAttn, ChannelMoE) muốn cải thiện.

---

## 4. Decoder — `Restormer_Decoder`

### Cấu trúc
```python
class Restormer_Decoder(nn.Module):
    self.reduce_channel  = Conv2d(128, 64, kernel=1)     # concat → dim/2
    self.encoder_level2  = [TransformerBlock × 4]        # same as Encoder
    self.output = Sequential(
        Conv2d(64, 32, kernel=3, padding=1),
        LeakyReLU(),
        Conv2d(32, 1, kernel=3, padding=1)
    )
    self.sigmoid = Sigmoid()
```

### Forward
```python
x = concat(f_B, f_D, dim=channel)                # [B, 128, H, W]
x = reduce_channel(x)                            # [B, 64, H, W]
x = TransformerBlock × 4 (x)                     # refine fused features
x = output_head(x)                               # [B, 1, H, W] (logit)
if input_img is not None:
    x = x + input_img                            # residual (IVF mode)
return sigmoid(x)
```

### Hai mode

| Mode | Behavior | Sử dụng |
|---|---|---|
| **IVF** (`input_img ≠ None`) | Có residual connection từ ảnh input | Infrared-Visible Fusion |
| **MIF** (`input_img = None`) | Không residual, output trực tiếp | Medical Image Fusion (CT/PET/SPECT-MRI) |

`test_MIF.py` luôn dùng MIF mode (`None`).

---

## 5. Hai pha huấn luyện

### Phase I (epoch 0–39) — Decomposition pretraining

**Forward** (per modality, không cross-modality):
```python
f_V_B, f_V_D, _ = Encoder(I_V)
f_I_B, f_I_D, _ = Encoder(I_I)
Î_V, _ = Decoder(I_V, f_V_B, f_V_D)              # reconstruct mỗi modality
Î_I, _ = Decoder(I_I, f_I_B, f_I_D)
```
*BaseFuseLayer/DetailFuseLayer KHÔNG được dùng — chưa có gradient.*

**Loss**:
```
L_I = α1·L_recon(I_V, Î_V) + α1·L_recon(I_I, Î_I)  ← reconstruction
    + α2·L_decomp                                  ← decomposition (CC formula)
    + α3·L_TV                                      ← total variation gradient
```
- `L_recon(X, X̂) = 5·SSIM(X, X̂) + MSE(X, X̂)`
- `L_decomp = (CC_D)² / (1.01 + CC_B)` — buộc Base correlated, Detail uncorrelated giữa 2 modality
- `L_TV = L1(∇I_V, ∇Î_V)` — preserve gradient của ảnh visible

**Update**: `optimizer1` (Encoder), `optimizer2` (Decoder).

### Phase II (epoch 40–119) — Fusion training

**Forward**:
```python
f_V_B, f_V_D, _ = Encoder(I_V)
f_I_B, f_I_D, _ = Encoder(I_I)
f_F_B = BaseFuseLayer(f_I_B + f_V_B)
f_F_D = DetailFuseLayer(f_I_D + f_V_D)
Î_F, _ = Decoder(I_V, f_F_B, f_F_D)               # fused output
```

**Loss**:
```
L_II = L_fusion(I_V, I_I, Î_F) + α4·L_decomp
```
- `L_fusion = L_int + L_grad`:
  - `L_int = ‖Î_F - max(I_V, I_I)‖²_F` — intensity loss với **max-pixel rule** (đây là chỗ Module B muốn thay)
  - `L_grad = ‖∇Î_F - max(∇I_V, ∇I_I)‖_1` — gradient preservation
- `L_decomp`: vẫn buộc Base/Detail decomp.

**Update**: TẤT CẢ 4 optimizers (Encoder, Decoder, BaseFuseLayer, DetailFuseLayer).

---

## 6. Hyperparameters paper

| | Paper text | Code release | Used in this project |
|---|---|---|---|
| Epochs total | 120 | 120 | 120 |
| Epoch gap (Phase I) | 40 | 40 | 40 |
| Batch size | **16** | **8** | 8 (P100 + AMP) |
| LR | 1e-4 | 1e-4 | 1e-4 |
| LR scheduler | StepLR γ=0.5/20 ep | same | same |
| α1 (MSE) | 1 | 1 | 1 |
| α2 (decomp Phase I) | 2 | 2 | 2 |
| α3 (TV) | **10** | **5** | 5 (theo code) |
| α4 (decomp Phase II) | 2 | 2 | 2 |
| Clip grad norm | 0.01 | 0.01 | 0.01 |
| Patch size | 128×128 | 128×128 | 128×128 |
| Stride | n/a | 200 (MSRS) | 64 (medical, ảnh nhỏ hơn) |
| GPU | 2× RTX 3090 | — | 1× P100 (Kaggle) |
| Dataset (IVF) | MSRS train 1083 | same | — |
| Dataset (MIF) | Harvard 286 (130 train + 20 val + 136 test) | — | same split logic |

---

## 7. Số parameters

| Component | Params | % | Notes |
|---|---|---|---|
| `Restormer_Encoder` | ~600K | ~50% | SFE (4 blocks) + BTE (1 block) + DCE (3 INN blocks) |
| `BaseFuseLayer` (BTE-like) | ~50K | ~4% | 1 transformer block dim=64, heads=8 |
| `DetailFuseLayer` (DCE-like) | ~30K | ~3% | 1 INN block |
| `Restormer_Decoder` | ~500K | ~42% | 4 transformer blocks + output conv |
| **TOTAL** | **~1.20M** | 100% | rất nhẹ so với DDFM/U2Fusion |

> **Reproducibility note**: Số chính xác xem `_ablation_stamp.json` của mỗi run.

---

## 8. Đặc điểm nổi bật

### Ưu điểm
1. **Lightweight**: 1.2M params, nhẹ hơn nhiều SOTA (DDFM 200M+, U2Fusion 6.4M).
2. **Restormer-based attention**: channel attention O(C²) thay vì spatial O((HW)²) → scale tốt với ảnh độ phân giải cao.
3. **Disentangled features**: tách Base (shared) vs Detail (modality-specific) explicit qua decomp loss.
4. **INN preserves info**: high-frequency detail không bị mất qua nén/decompose.
5. **2-phase training**: pretraining (Phase I) ổn định features trước khi học fusion (Phase II).

### Điểm yếu (đã phân tích trong project này)
1. **Phép cộng `f_I + f_V` đơn giản** ở fuse layers — không tận dụng được modality-specific weighting. → **Module A** đề xuất Gated/CrossAttn/ChannelMoE.
2. **Max-pixel rule trong `L_int^II`** — tạo boundary artifact ở vùng pixel intensity trái chiều. → **Module B** đề xuất Mean/Saliency/Learnable.
3. **Linear Pearson CC trong `L_decomp`** — chỉ capture linear dependence, có thể bỏ sót nonlinear relationships giữa Base/Detail. → **Module C** đề xuất CKA/HSIC/Orthogonality.
4. **Sample size training nhỏ** cho MIF (chỉ 130 train pairs) — khó học deep representations robust.

---

## 9. Forward call summary (test/inference)

```python
# Load checkpoint
ckpt = torch.load('CDDFuse_MIF.pth')
encoder.load_state_dict(ckpt['DIDF_Encoder'])
decoder.load_state_dict(ckpt['DIDF_Decoder'])
base_fuse.load_state_dict(ckpt['BaseFuseLayer'])
detail_fuse.load_state_dict(ckpt['DetailFuseLayer'])

# Inference
with torch.no_grad():
    f_V_B, f_V_D, _ = encoder(I_V)
    f_I_B, f_I_D, _ = encoder(I_I)
    f_F_B = base_fuse(f_V_B + f_I_B)
    f_F_D = detail_fuse(f_V_D + f_I_D)
    fused, _ = decoder(None, f_F_B, f_F_D)         # MIF: None
    # Min-max normalize
    fused = (fused - fused.min()) / (fused.max() - fused.min())
```

---

## 10. Tham chiếu code

| Component | File | Line range |
|---|---|---|
| Encoder | `net.py` | 336–362 |
| Decoder | `net.py` | 364–395 |
| BaseFeatureExtraction (BTE) | `net.py` | 109–124 |
| DetailFeatureExtraction (DCE) | `net.py` | 167–176 |
| DetailNode (INN block) | `net.py` | 148–165 |
| TransformerBlock (Restormer) | `net.py` | 306–318 |
| MDTA Attention | `net.py` | 266–304 |
| GDFN FeedForward | `net.py` | 241–264 |
| LayerNorm | `net.py` | 192–238 |
| Fusionloss + cc() | `utils/loss.py` | — |
| train.py 2-phase loop | `train.py` | 100–199 |

---

## 11. Tham khảo

- Zhao et al., "CDDFuse: Correlation-Driven Dual-Branch Feature Decomposition for Multi-Modality Image Fusion", **CVPR 2023**. [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_CDDFuse_Correlation-Driven_Dual-Branch_Feature_Decomposition_for_Multi-Modality_Image_Fusion_CVPR_2023_paper.pdf)
- Zamir et al., "Restormer: Efficient Transformer for High-Resolution Image Restoration", **CVPR 2022** (kiến trúc backbone).
- Dinh et al., "Density estimation using Real NVP", **ICLR 2017** (INN blocks).
