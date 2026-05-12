# Phân tích chi tiết kiến trúc CDDFuse & hướng cải tiến

> **Mục đích**: Tài liệu kỹ thuật phục vụ luận văn — phân tích sâu CDDFuse để xây dựng đề xuất cải tiến cho medical image fusion.
> **Nguồn code**: [models/MMIF-CDDFuse/](../models/MMIF-CDDFuse/)
> **Paper gốc**: Zhao et al., "CDDFuse: Correlation-Driven Dual-Branch Feature Decomposition for Multi-Modality Image Fusion", **CVPR 2023** ([Paper PDF](../Paper/Zhao_CDDFuse_Correlation-Driven_Dual-Branch_Feature_Decomposition_for_Multi-Modality_Image_Fusion_CVPR_2023_paper.pdf) · [arXiv 2211.14461](https://arxiv.org/abs/2211.14461) · [GitHub](https://github.com/Zhaozixiang1228/MMIF-CDDFuse))
>
> **Cấu trúc**: §1-11 = phân tích từ source code + tổng hợp. **§12 = chi tiết chính xác từ paper gốc** (đối chiếu, hiệu chỉnh các mô tả ở §1-11). §13 = tổng kết.

---

## 1. Tổng quan

| | |
|---|---|
| **Tên đầy đủ** | Correlation-Driven Dual-branch Feature Decomposition Fusion |
| **Năm xuất bản** | 2023 (CVPR) |
| **Nhiệm vụ** | Multi-modality image fusion (IR-VIS, MIF: medical) |
| **Backbone** | Restormer-style Transformer (MDTA + GDFN) + INN |
| **Số tham số** | ~1.2M (encoder + decoder) |
| **Resolution** | Full-resolution (no downsampling) |
| **Training** | 2-stage: AE pretraining (40 epochs) → Fusion training (80 epochs) |
| **Performance trong báo cáo này** | Composite Z = +0.93 (rank #2 / 22), n=72 |

---

## 2. Triết lý cốt lõi (Motivation)

### Vấn đề CDDFuse muốn giải

Fusion 2 ảnh từ modality khác nhau gặp 3 vấn đề:

1. **Trùng lặp thông tin** giữa 2 source (cùng anatomy/structure) — gọi là *common features*
2. **Bổ sung thông tin** (mỗi modality có đặc trưng riêng) — gọi là *unique features*
3. **Cách fuse 2 loại này KHÁC NHAU** nhưng các phương pháp cũ đối xử như nhau

### Giải pháp của CDDFuse

> **Decompose features thành 2 thành phần tần số rồi fuse riêng:**
> - **Base feature** (low-frequency) ← chứa structure/anatomy chung → **CORRELATED giữa A và B**
> - **Detail feature** (high-frequency) ← chứa edges/texture/modality-specific → **DECORRELATED**

Logic:
- Vì cùng chụp cùng vùng cơ thể → MRI và CT/PET share anatomy → base correlated
- Vì cơ chế chụp khác nhau → detail texture khác → detail decorrelated
- **Train encoder để PHẢN ÁNH đúng tính chất này** (correlation-driven loss)

---

## 3. Kiến trúc chi tiết

### 3.1 Sơ đồ tổng

```
┌─────────────────────────────────────────────────────────────┐
│                    SHARED ENCODER (E)                        │
│                                                              │
│  [Conv 3×3] → [TransformerBlock × 4]  ←  shared trunk       │
│                       │                                      │
│              ┌────────┴────────┐                            │
│              ▼                 ▼                             │
│       Base Branch         Detail Branch                      │
│       (1 TR block)        (3 INN coupling)                   │
│              │                 │                             │
│              ▼                 ▼                             │
│        base_feat          detail_feat                        │
└─────────────────────────────────────────────────────────────┘

      Encode A ─► (base_A, detail_A)
      Encode B ─► (base_B, detail_B)
                    │           │
                    ▼           ▼
               BaseFuseLayer  DetailFuseLayer    ← train Phase II
                    │           │
                    ▼           ▼
              base_fused   detail_fused

┌─────────────────────────────────────────────────────────────┐
│                     DECODER (D)                              │
│                                                              │
│  concat → 1×1 Conv (reduce) → [TransformerBlock × 4]         │
│         → 3×3 Conv → 3×3 Conv → Sigmoid                      │
│                       │                                      │
│              + input_residual                                │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
                  Fused output
```

### 3.2 Module: TransformerBlock (Restormer-style)

Mỗi block gồm 2 sub-layer:

#### a) MDTA — Multi-Dconv Head Transposed Self-Attention

```python
qkv = Conv1×1(x) → Conv3×3 (depthwise)        # tăng locality
q, k, v = chunk(qkv, 3)
q, k normalize theo channel
attn = (q @ k.T) * temperature                # CHANNEL attention!
out = attn @ v
```

**Khác biệt với attention thông thường**:
- Attention theo **channel** (C × C matrix) thay vì **spatial** (HW × HW)
- Complexity: O(C²) thay vì O((HW)²) → nhẹ hơn rất nhiều với ảnh y khoa cao res
- Vẫn capture global dependency vì mỗi channel = global feature map

#### b) GDFN — Gated-Dconv Feed-Forward Network

```python
x = Conv1×1(x)               # expand 2× channels
x1, x2 = dwconv(x).chunk(2)
x = GELU(x1) * x2            # gating mechanism
x = Conv1×1(x)               # back to original dim
```

**Đặc điểm**: Gating chọn thông tin nào pass through → tăng expressive power so với MLP thường.

### 3.3 Module: BaseFeatureExtraction

```python
out = x + Attention(LayerNorm(x))   # spatial attention, dim=64
out = out + MLP(LayerNorm(out))
```

→ 1 transformer block với **spatial attention** (khác MDTA — đây dùng spatial vì capture LOW-frequency structure cần spatial context)

### 3.4 Module: DetailNode (INN — Invertible Neural Network)

Coupling layer kiểu RealNVP/NICE:

```python
def forward(z1, z2):
    z1, z2 = separate(shuffle_conv(concat(z1, z2)))
    z2 = z2 + φ(z1)                    # additive coupling
    z1 = z1 * exp(ρ(z2)) + η(z2)       # affine coupling
    return z1, z2
```

Trong đó φ, ρ, η là **InvertedResidualBlock** (MobileNetV2-style).

**Tại sao dùng INN cho detail?**
- INN bảo toàn information (invertible) → high-frequency details không bị mất qua nhiều layer
- Phù hợp với detail (cần preserve edges, không tolerable với info loss)

### 3.5 BaseFuseLayer & DetailFuseLayer

Đây là **modules dùng KHI INFER** để fuse 2 base/detail features:

```python
BaseFuseLayer = BaseFeatureExtraction(dim=64, num_heads=8)   # 1 transformer block
DetailFuseLayer = DetailFeatureExtraction(num_layers=1)      # 1 INN coupling
```

**Inference trong code** ([test_MIF.py](../models/MMIF-CDDFuse/test_MIF.py)):
```python
feature_F_B = BaseFuseLayer(feature_A_B + feature_B_B)   # fuse base
feature_F_D = DetailFuseLayer(feature_A_D + feature_B_D) # fuse detail
data_Fuse, _ = Decoder(input_A, feature_F_B, feature_F_D)
```

→ **Fusion rule**: cộng features rồi qua 1 transformer/INN block để "tinh chỉnh".
→ Đơn giản nhưng hiệu quả vì features đã được decompose đúng.

---

## 4. Loss functions

### 4.1 Phase I (epoch 0–39): AutoEncoder pretraining

```
L_phase1 = α₁ · L_VIS + α₂ · L_IR + α₃ · L_decomp + α₄ · L_grad
```

Trong đó:

| Component | Công thức | Vai trò |
|---|---|---|
| L_VIS | `5·SSIM(VIS, VIŜ) + MSE(VIS, VIŜ)` | Reconstruct ảnh visible/MRI |
| L_IR | `5·SSIM(IR, IR̂) + MSE(IR, IR̂)` | Reconstruct ảnh infrared/CT |
| **L_decomp** | `(cc_D)² / (1.01 + cc_B)` | **CORE INSIGHT**: maximize cc_B, minimize cc_D |
| L_grad | `L1(SpatialGradient(VIS), SpatialGradient(VIŜ))` | Bảo toàn edge khi reconstruct |

**Hệ số (default)**: α₁ = α₂ = 1.0, α₃ = 2.0, α₄ = 5.0

**`cc` = Pearson correlation coefficient** giữa 2 feature maps.

L_decomp = (cc_D)² / (1.01 + cc_B):
- Tử số (cc_D)² → muốn = 0 (decorrelated detail)
- Mẫu số (1.01 + cc_B) → muốn lớn (correlated base)
- Tổng: minimize tử / maximize mẫu

### 4.2 Phase II (epoch 40–119): Fusion training

```
L_phase2 = L_fusion + α₃ · L_decomp
```

`L_fusion` từ [utils/loss.py — Fusionloss class](../models/MMIF-CDDFuse/utils/loss.py) gồm:
- **L_intensity**: pixel-level loss với max-rule (`max(VIS, IR)` làm target)
- **L_grad**: gradient consistency
- (No SSIM ở phase II)

L_decomp giữ nguyên từ Phase I.

### 4.3 Hyperparams training

| Param | Value | Ghi chú |
|---|---|---|
| Optimizer | Adam (4 separate, mỗi component 1 optimizer) | |
| LR | 1e-4 → 1e-6 (StepLR, gamma=0.5, step=20) | |
| Batch size | 8 | |
| Crop size | 128×128 | |
| Total epochs | 120 (40 phase I + 80 phase II) | |
| Grad clip | 0.01 | rất conservative |

---

## 5. Inference (Fusion rule)

```python
# 1. Encode independently
B_A, D_A, _ = Encoder(img_A)
B_B, D_B, _ = Encoder(img_B)

# 2. Fuse (sum then refine)
B_F = BaseFuseLayer(B_A + B_B)
D_F = DetailFuseLayer(D_A + D_B)

# 3. Decode
fused, _ = Decoder(img_A, B_F, D_F)   # img_A dùng làm residual reference
```

**Key observations**:
- Fusion rule = **simple sum + learned refinement** (không phải max/avg fix sẵn)
- Decoder lấy `img_A` làm residual → output có "anchor" theo ảnh A
- → Output bias nhẹ về modality A (thường là MRI hoặc VIS)

---

## 6. Điểm mạnh

| | |
|---|---|
| ✅ **Decomposition có ý nghĩa** | Base/Detail decomposition có lý thuyết rõ (frequency-based, correlation-driven) |
| ✅ **Restormer backbone** | MDTA hiệu quả, scale tốt với resolution lớn |
| ✅ **2-stage training ổn định** | Pretrain encoder làm AE → fusion ít overfitting |
| ✅ **Multi-task** | Train 1 lần dùng cho IR-VIS, medical, MEF, MFF |
| ✅ **Output sharpness** | Nhờ INN bảo toàn detail, residual connection với input |
| ✅ **Code clean** | Modular, dễ extend |

---

## 7. Điểm yếu / Hạn chế

| Hạn chế | Chi tiết | Tác động |
|---|---|---|
| **a) No multi-scale** | Single-resolution, không downsample → miss large-scale context | Yếu với lesion lớn hoặc structure nested |
| **b) Decomposition cứng** | Chỉ 2 nhánh (base vs detail), boundary không learnable | Mid-frequency features bị nhập nhằng |
| **c) Fusion rule ngây thơ** | A + B rồi qua 1 block — không content-aware | Khi A và B contrast khác xa, sum đơn giản gây washout |
| **d) Color không xử lý native** | Trong MIF: chỉ Y channel, Cb/Cr re-merge từ source | Không tận dụng được thông tin color của PET/SPECT |
| **e) Encoder shared** | Dùng cùng E cho A và B → buộc 2 modality về cùng feature space | Mất modality-specific patterns |
| **f) Loss đơn giản** | MSE + SSIM + L_decomp; không có anatomical-aware | Có thể fuse "đẹp pixel" nhưng sai structure y khoa |
| **g) Không có uncertainty** | Output deterministic, không expose confidence map | Khó dùng trong clinical decision support |
| **h) Train trên IR-VIS** | Pretrained chính trên MSRS dataset (IR-VIS) | Domain gap khi áp medical |

---

## 8. Hướng cải tiến (categorized by novelty + implementation difficulty)

### 8.1 🔴 Nhóm A — Architectural innovations (high novelty)

#### A1. Hybrid Mamba-Transformer Encoder

**Motivation**: MFS-Fusion (Mamba-based) đang #1 SSIM. Mamba có linear complexity O(N) vs Transformer O(N²) → tốc độ + chất lượng cùng cải thiện.

**Đề xuất**: Trong `Restormer_Encoder`, thay 4 TransformerBlock bằng:
- 2 TransformerBlock (giữ global attention)
- 2 MambaBlock (long-range dependency với linear cost)

```python
class HybridEncoder(nn.Module):
    def __init__(self, dim=64):
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads=8, ffn_expansion_factor=2, ...),  # block 1
            MambaBlock(dim),                                                    # block 2
            TransformerBlock(dim, num_heads=8, ffn_expansion_factor=2, ...),  # block 3
            MambaBlock(dim),                                                    # block 4
        ])
```

**Khó**: Trung bình. Cần tích hợp `mamba-ssm` package.

#### A2. 3-band Decomposition (Base / Mid / Detail)

**Motivation**: 2 nhánh chia thô. Y khoa có structures ở 3 thang (anatomical contour / organ texture / lesion edge).

**Đề xuất**: Thêm `MidFeatureExtraction` branch song song với Base & Detail:

```python
class TriBandEncoder(nn.Module):
    def __init__(self):
        ...
        self.baseFeature = BaseFeatureExtraction(...)        # low-freq
        self.midFeature  = MidFeatureExtraction(...)         # NEW: mid-freq
        self.detailFeature = DetailFeatureExtraction(...)    # high-freq
    
    def forward(self, x):
        feat = self.encoder_level1(self.patch_embed(x))
        return self.baseFeature(feat), self.midFeature(feat), self.detailFeature(feat)
```

**MidFeatureExtraction** có thể dùng:
- Wavelet decomposition (DWT) để đảm bảo lý thuyết frequency-based
- Hoặc 1 transformer block với attention scale khác (window-based)

**Loss thay đổi**: L_decomp mở rộng cho 3 cặp:
```
L_decomp_3band = (cc_D)²/(1.01+cc_B) + (cc_M)²/(1.01+cc_B) + λ·cc_M·cc_D
```

**Khó**: Khó (cần thiết kế lại loss + tune hyperparam). **High novelty**.

#### A3. Modality-Aware Encoder với Cross-Attention

**Motivation**: Encoder shared bỏ qua modality-specific patterns.

**Đề xuất**: 2 encoder lite + cross-attention khi merge:

```python
class CrossModalEncoder(nn.Module):
    def __init__(self):
        self.E_anat = MiniEncoder()           # cho MRI (anatomical)
        self.E_func = MiniEncoder()           # cho PET/CT/SPECT (functional)
        self.cross_attn = CrossAttention(dim=64)
    
    def forward(self, x_anat, x_func):
        f_a = self.E_anat(x_anat)
        f_f = self.E_func(x_func)
        f_a, f_f = self.cross_attn(f_a, f_f)  # mutual conditioning
        return f_a, f_f
```

**Khó**: Trung bình.

### 8.2 🟢 Nhóm B — Loss & training (quick wins)

#### B1. Anatomical-aware perceptual loss

**Motivation**: Loss hiện tại pixel-level → không đảm bảo structure y khoa.

**Đề xuất**: Thêm L_anat = L_perceptual với pretrained medical model:

```python
class MedicalPerceptualLoss(nn.Module):
    def __init__(self):
        self.backbone = load_pretrained('biomedclip')  # hoặc MedSAM, RadiologyBert
        for p in self.backbone.parameters(): p.requires_grad = False
    
    def forward(self, fused, target_anat):
        f_fused = self.backbone(fused)
        f_target = self.backbone(target_anat)
        return F.mse_loss(f_fused, f_target)

L_total = L_fusion + α·L_decomp + β·L_anat
```

**Khó**: Dễ (chỉ thêm loss, không đụng architecture).

#### B2. Lesion-region weighted loss

```python
def lesion_weighted_loss(fused, source, lesion_mask):
    # Weight loss cao hơn ở vùng lesion
    weight = 1.0 + 5.0 * lesion_mask
    return (weight * (fused - source).abs()).mean()
```

Lesion mask có thể từ pretrained TotalSegmentator hoặc nnUNet.

**Khó**: Dễ-trung bình (cần lesion mask).

#### B3. Frequency-domain loss

Thêm L_freq = MSE giữa FFT(fused) và FFT(source):

```python
def freq_loss(fused, source):
    F_fused = torch.fft.fft2(fused)
    F_source = torch.fft.fft2(source)
    return F.l1_loss(F_fused.abs(), F_source.abs())
```

→ Buộc fused giữ phổ tần số → ít artifact.

**Khó**: Rất dễ.

### 8.3 🔵 Nhóm C — Inference improvements (no retrain)

#### C1. Learnable fusion rule (replace `A + B`)

**Motivation**: Hiện tại fusion = sum thuần.

**Đề xuất**: Train 1 small MLP làm fusion gate:

```python
class GatedFusion(nn.Module):
    def __init__(self, dim):
        self.gate = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1), nn.GELU(),
            nn.Conv2d(dim, dim, 1), nn.Sigmoid()
        )
    def forward(self, f_a, f_b):
        g = self.gate(torch.cat([f_a, f_b], dim=1))
        return g * f_a + (1 - g) * f_b
```

**Khó**: Dễ.

#### C2. Test-Time Augmentation (TTA)

```python
def fuse_tta(model, A, B):
    outs = []
    for tf in [identity, hflip, vflip, rot90, rot180]:
        outs.append(tf.inverse(model(tf(A), tf(B))))
    return torch.stack(outs).mean(0)
```

**Lợi**: +1-2% SSIM gần như miễn phí.

**Khó**: Rất dễ.

#### C3. Knowledge Distillation từ MFS-Fusion

Dùng MFS-Fusion (top-1 SSIM) làm teacher:

```
L_distill = α·L_original + β·MSE(fused_student, fused_teacher)
```

→ Combine: tốc độ CDDFuse + chất lượng MFS-Fusion.

**Khó**: Trung bình.

### 8.4 🟡 Nhóm D — Color processing (medical-specific)

PET/SPECT là RGB (color-coded functional info). CDDFuse hiện chỉ xử lý Y channel.

**Đề xuất D1**: Joint Y-CbCr fusion:

```python
class ColorAwareCDDFuse(nn.Module):
    def __init__(self):
        self.E_y = Restormer_Encoder(inp_channels=1)
        self.E_c = Restormer_Encoder(inp_channels=2)   # Cb, Cr
        self.cross_attn_yc = CrossAttention(dim=64)
        ...
```

**Khó**: Trung bình.

---

## 9. Đề xuất CỤ THỂ cho luận văn

### 9.1 Đề xuất chính

**Tên**: **CDDFuse-Medical: Hybrid Mamba-Transformer with Tri-band Decomposition and Anatomical-Aware Loss**

Combine 3 hướng:
1. **A1 (Hybrid Mamba-Transformer)** — modern backbone, novelty cao
2. **A2 (3-band decomposition)** — architectural contribution, có lý thuyết
3. **B1 (Anatomical-aware loss)** — medical-specific, defense dễ

### 9.2 Vì sao combo này mạnh?

| Tiêu chí | Đáp ứng |
|---|---|
| **Novelty** | 3 contribution trong 1 paper |
| **Đo lường được** | Có thể ablation study từng component |
| **Trending** | Mamba (2024), Multi-band (cổ điển nhưng ít dùng cho fusion), Medical-aware (đang nóng) |
| **Defense** | Mỗi cải tiến có lý do KH rõ ràng |
| **Implementation** | Có baseline (CDDFuse), không phải build from scratch |

### 9.3 Roadmap thực hiện (gợi ý 3 tháng)

| Tuần | Việc |
|---|---|
| 1-2 | Setup environment, reproduce CDDFuse baseline trên Harvard medical dataset, đo metric |
| 3-4 | Implement A1 (Hybrid Mamba), train, so sánh với baseline |
| 5-6 | Implement A2 (3-band), train, ablation |
| 7-8 | Implement B1 (Anatomical loss), tune hyperparam |
| 9 | Combine 3 → train full model |
| 10 | Ablation study chi tiết (turn off từng component) |
| 11 | Visualization, qualitative analysis |
| 12 | Viết luận văn, chuẩn bị defense |

### 9.4 Baseline & target

| Metric | CDDFuse baseline (n=72) | Target sau cải tiến |
|---|---|---|
| SSIM | 1.4387 | **> 1.46** (đuổi MFS-Fusion 1.4479) |
| QG | 0.7023 | > 0.75 |
| QY | 0.9559 | > 0.97 |
| QCB | 0.8577 | > 0.88 |
| Composite Z | +0.93 | **> +1.10** (vượt MM-Net-Fusion) |

### 9.5 Ablation study cần có

| Variant | Components |
|---|---|
| A | Baseline (CDDFuse) |
| B | A + Hybrid Mamba |
| C | A + 3-band decomposition |
| D | A + Anatomical loss |
| E | A + B + C |
| F | A + B + D |
| G | A + C + D |
| H | A + B + C + D (full proposed) |

→ Bảng 8 dòng × n metric → reviewer thích.

---

## 10. Câu hỏi reviewer/hội đồng có thể hỏi

| Câu hỏi | Câu trả lời gợi ý |
|---|---|
| *Tại sao chọn CDDFuse mà không phải MFS-Fusion (top-1)?* | MFS-Fusion runtime 50 phút/modality trên CPU — không production-ready. CDDFuse cân bằng hơn, code clean dễ extend, có 2-stage training ổn định. |
| *Mamba có thực sự hiệu quả cho fusion?* | MFS-Fusion (Mamba-based) #1 SSIM trong báo cáo; Mamba đã chứng minh competitive với Transformer ở image tasks 2024. |
| *3-band decomposition có cơ sở lý thuyết?* | Wavelet multi-resolution analysis, image pyramid theory. Mid-band capture organ-level texture mà 2-band CDDFuse miss. |
| *Anatomical-aware loss với pretrained model có overfitting?* | Pretrained model frozen, chỉ dùng làm feature extractor → không overfit. Tương tự perceptual loss VGG đã được dùng rộng rãi. |
| *Tại sao improvement chỉ vài %?* | Trong fusion, top-tier model đã sát nhau. 1% SSIM tương đương 0.1σ z-score → có ý nghĩa thống kê khi p-value < 0.05. |

---

## 11. Tài liệu tham khảo gợi ý

### Paper gốc và mở rộng
1. **Zhao et al.** "CDDFuse" — CVPR 2023 [paper gốc]
2. **Zhao et al.** "DDFM: Denoising Diffusion Model for Multi-modality Image Fusion" — ICCV 2023 [cùng nhóm]
3. **Zamir et al.** "Restormer: Efficient Transformer for High-Resolution Image Restoration" — CVPR 2022 [Restormer block]

### Mamba-based vision
4. **Zhu et al.** "Vision Mamba" — 2024
5. **Liu et al.** "VMamba" — 2024

### Medical fusion related
6. **Tang et al.** "MFS-Fusion: Mamba-based Spatial Frequency Fusion" — 2025 [top-1 trong báo cáo]
7. **MM-Net-Fusion** — MixFormer-based, medical specific [đã đánh giá]

### INN / coupling
8. **Dinh et al.** "Real NVP" — 2017 [INN coupling layer]

### Medical foundation models
9. **BioMedCLIP** — 2023
10. **MedSAM** — 2024
11. **TotalSegmentator** — 2023

---

## 12. Phụ lục — Chi tiết chuẩn từ paper gốc (CVPR 2023)

> **Quan trọng**: Phần 1-11 ở trên là phân tích từ source code và tổng hợp tôi tự viết. Phần 12 này được trích & đối chiếu trực tiếp từ paper gốc Zhao et al., CVPR 2023 (đã có ở [Paper/](../Paper/)). Có vài chi tiết tôi mô tả ở §3 chưa hoàn toàn đúng — mục này hiệu chỉnh.

### 12.1 Đóng góp chính (4 contributions từ paper, §1)

Trích nguyên văn:

1. **Dual-branch Transformer-CNN framework** for extracting and fusing global and local features → reflects modality-specific & shared features tốt hơn.
2. **Refine CNN and Transformer blocks** for MMIF: là **first** dùng:
   - **INN blocks** cho lossless information transmission
   - **LT blocks** cho trading-off fusion quality vs computational cost
3. **Correlation-driven decomposition loss** để enforce modality shared/specific decomposition.
4. **SOTA performance** trên IVF + MIF + downstream MM detection/segmentation.

### 12.2 Kiến trúc CHÍNH XÁC (paper Figure 2)

Tôi đã mô tả chưa chuẩn ở §3. Kiến trúc thực tế **3 thành phần encoder** (không phải 2):

```
                          ┌─────────────────────────────┐
                          │   Encoder (3 components)    │
                          │                             │
  Input I ─►              │  ┌──────────────────────┐   │
  (infrared/CT/PET)       │  │ Share Feature Encoder│   │
                          │  │   SFE = S(·)         │   │  ← 4 Restormer blocks
                          │  │   [Restormer-based]  │   │
                          │  └──────────┬───────────┘   │
                          │             │ shallow φ^S    │
                          │             ├──────────────┐ │
                          │             ▼              ▼ │
                          │  ┌──────────────────┐  ┌──────────────┐ │
                          │  │ Base Transformer │  │ Detail CNN   │ │
                          │  │ Encoder (BTE)    │  │ Encoder (DCE)│ │
                          │  │ B(·) [LT blocks] │  │ D(·) [INN]   │ │
                          │  └──────────────────┘  └──────────────┘ │
                          │       │ φ^B = base       │ φ^D = detail │
                          └───────┼──────────────────┼──────────────┘
                                  │                  │
                          (BTE + DCE = "Long-short Range Encoder")
```

**3 encoder modules**:

| Module | Ký hiệu | Backbone | Vai trò |
|---|---|---|---|
| **SFE** (Share Feature Encoder) | S(·) | **Restormer blocks ×4**, dim=64, 8 heads | Trích shallow features từ ảnh đầu vào — global shallow representation |
| **BTE** (Base Transformer Encoder) | B(·) | **Lite Transformer (LT) blocks**, dim=64, 8 heads | Trích **low-frequency BASE** (long-range, modality-shared) |
| **DCE** (Detail CNN Encoder) | D(·) | **INN blocks** với BRB (bottleneck residual block) làm sub-mapping | Trích **high-frequency DETAIL** (local, modality-specific, lossless) |

→ **BTE + DCE** được gọi là **Long-short Range Encoder** (long-range qua transformer, short-range qua CNN/INN).

⚠️ **Khác biệt với mô tả ở §3 của tôi**:
- Tôi gọi nhầm `BaseFeatureExtraction` (1 transformer block) là core của BTE. Thực tế paper dùng **LT blocks**, không phải Restormer Transformer Block như tôi mô tả.
- Tôi mô tả encoder shared 4 transformer block + 2 nhánh nối tiếp. Thực tế là **3 encoder song song-tuần tự** (SFE → BTE & DCE).
- File [net.py](../models/MMIF-CDDFuse/net.py) trong code có module `BaseFeatureExtraction` mà cấu trúc nhìn giống transformer block thông thường — đây thực ra là **simplified LT block** trong implementation, paper mô tả đầy đủ hơn.

### 12.3 LT (Lite Transformer) block — chi tiết

LT do Wu et al. (ICLR 2020) đề xuất:
- **Long-Short Range Attention (LSRA)** chia head thành 2 nhóm:
  - Một nửa: standard self-attention (global)
  - Một nửa: depthwise convolution (local)
- **Flattened Feed-Forward Network (FFN)**: làm phẳng bottleneck → giảm tham số nhưng vẫn duy trì performance

**Lý do paper chọn LT thay vì Restormer cho BTE**: cân bằng quality vs compute. Restormer mạnh nhưng nặng → SFE+Decoder (cần expressive power) dùng Restormer; BTE chỉ cần extract base features → LT nhẹ hơn.

### 12.4 INN block — chi tiết (paper §3.2)

Mỗi INN có K invertible layers. Trong layer thứ k (k = 1, ..., K):

```
φ^S_{I,k+1}[c+1:C] = φ^S_{I,k}[c+1:C] + I₁(φ^S_{I,k}[1:c])

φ^S_{I,k+1}[1:c]   = φ^S_{I,k}[1:c] ⊙ exp(I₂(φ^S_{I,k+1}[c+1:C]))
                                          + I₃(φ^S_{I,k+1}[c+1:C])

φ^S_{I,k+1} = CAT{φ^S_{I,k+1}[1:c], φ^S_{I,k+1}[c+1:C]}
```

trong đó:
- ⊙ = Hadamard product
- φ^S_{I,k}[1:c] = c kênh đầu của input layer thứ k
- CAT = channel concatenation
- I_1, I_2, I_3 = arbitrary mapping functions → paper dùng **BRB (Bottleneck Residual Block)** từ MobileNetV2

**Tại sao chọn INN cho detail?**
- INN có **invertibility by design** → ngăn information loss qua propagation
- "Aligns with our goal of preserving high-frequency features in the fused images" (paper §1)

### 12.5 Loss functions — công thức CHÍNH XÁC từ paper

#### Stage I (epochs 0 → 39):

```
L_total^I = L_ir + α₁·L_vis + α₂·L_decomp                  (Eq.7)
```

Reconstruction loss cho từng modality:
```
L_ir = L_int^I(I, Î) + μ·L_SSIM(I, Î)                       (Eq.8)
L_int^I = ||I - Î||₂²
L_SSIM = 1 - SSIM(·)
```

(L_vis tương tự với V và V̂.)

Decomposition loss:
```
L_decomp = (CC_D)² / (CC_B + ε)                             (Eq.9)
        = [CC(φ^D_I, φ^D_V)]² / [CC(φ^B_I, φ^B_V) + ε]
```

với ε = 1.01, CC = Pearson correlation.

**Logic** (paper §3.5):
- Numerator (CC_D)² → 0: detail features của I và V uncorrelated (modality-specific)
- Denominator CC_B → tăng: base features highly correlated (modality-shared)
- Empirically, CC_D → 0 và CC_B → lớn dần qua training

#### Stage II (epochs 40 → 119):

```
L_total^II = L_int^II + α₃·L_grad + α₄·L_decomp             (Eq.10)
```

Intensity loss với **max-rule selection**:
```
L_int^II = (1/HW) · ||I_f - max(I_ir, I_vis)||₁
```

Gradient loss với Sobel operator ∇:
```
L_grad = (1/HW) · ||∇I_f - max(|∇I_ir|, |∇I_vis|)||₁
```

→ **Bias output về modality có pixel/gradient mạnh hơn** ở mỗi vị trí.

### 12.6 Hyperparameters CHÍNH XÁC từ paper §4.1

| Param | Paper value | Note |
|---|---|---|
| GPUs | **2× NVIDIA RTX 3090** | (đáng kể: paper dùng 2 GPU, không phải 1) |
| Batch size | **16** | (train.py local có thể là 8 — config khác) |
| Epoch total | 120 (40 + 80) | Stage I: 40, Stage II: 80 |
| Crop size | 128×128 | random crop |
| Optimizer | Adam | LR=1e-4, decay 0.5× mỗi 20 epochs |
| **α₁, α₂** (Stage I weights) | **2, 2** | reconstruction + decomp |
| **α₃, α₄** (Stage II weights) | **10, 2** | grad + decomp |
| μ (SSIM weight in Stage I) | (paper không nói rõ; train.py = 5) | |
| Restormer blocks trong SFE | **4 blocks** | dim=64, 8 attention heads |
| LT block trong BTE | dim=64, **8 attention heads** | |
| Decoder | Same config as encoder (Restormer) | |

### 12.7 Datasets

#### IVF (Infrared-Visible Fusion):
- **Train**: MSRS (1083 pairs)
- **Validation**: RoadScene (50 pairs)
- **Test**: MSRS (361 pairs), RoadScene (50 pairs), TNO (25 pairs)
- Note: **không** fine-tune trên RoadScene/TNO → test generalization

#### MIF (Medical Image Fusion):
- Source: **286 pairs** từ Harvard Medical website
- Split: 130 train / 20 val
- **Test**:
  - 21 pairs MRI-CT
  - 42 pairs MRI-PET
  - 73 pairs MRI-SPECT

→ **2 setup được test**:
1. **CDDFuse** (no fine-tune): chỉ train trên IVF, test trên MIF → đo *generalization*
2. **CDDFuse\*** (fine-tune): train từ đầu trên MIF dataset → đo *upper bound*

### 12.8 Ablation Study từ paper (Table 2, MSRS test set)

Paper kết luận **mọi component đều quan trọng**. Kết quả:

| Configuration | EN ↑ | SD ↑ | VIF ↑ | SSIM ↑ |
|---|---|---|---|---|
| **CDDFuse (Ours)** | **6.70** | **43.38** | **1.05** | **1.00** |
| I — Division → Subtraction trong L_decomp (`(L_CC^D)² - L_CC^B`) | 6.55 | 42.20 | 0.98 | 1.00 |
| II — w/o L_decomp | 6.19 | 36.49 | 0.96 | 0.97 |
| III — LT → INN trong BTE | 6.47 | 41.39 | 1.00 | 0.97 |
| IV — INN → LT trong DCE | 6.56 | 42.18 | 1.00 | 0.99 |
| V — INN → CNN (BRB) trong DCE | 6.54 | 42.10 | 0.98 | 0.98 |
| VI — w/o two-stage training | 6.28 | 38.42 | 0.97 | 0.99 |

**Insight quan trọng từ ablation**:

| Component | Tác động khi remove | Insight |
|---|---|---|
| **L_decomp** | EN giảm 7.6%, SSIM giảm 3% | **Critical**. Decomposition loss thực sự enforce shared/specific separation |
| **Two-stage training** | EN giảm 6.3% | Cần thiết — không thể train end-to-end one-shot |
| **INN trong DCE** | EN giảm 2.1%, SSIM giảm 1% | INN preserve detail tốt hơn LT/CNN |
| **LT trong BTE** | EN giảm 3.4%, SSIM giảm 3% | LT phù hợp hơn INN cho base feature |
| **L_decomp formulation** | Division (`(CC_D)²/(CC_B+ε)`) > Subtraction (`(CC_D)² - CC_B`) | Form division gây áp lực mạnh hơn để CC_B tăng và CC_D giảm |

→ **Ý nghĩa cho cải tiến**: nếu thay LT/INN → có thể bị penalty. Nếu thay L_decomp → cần verify carefully (không phải mọi formulation đều work).

### 12.9 Performance trên Medical Image Fusion (Table 5 paper)

| Dataset | Method | EN | SD | SF | MI | SCD | VIF | Q^AB/F | SSIM |
|---|---|---|---|---|---|---|---|---|---|
| **MRI-CT** | TarD | 4.75 | 61.14 | 28.38 | 1.94 | 0.81 | 0.32 | 0.35 | 0.61 |
| | RFN | 5.30 | 52.95 | 33.42 | 1.98 | 0.58 | 0.33 | 0.52 | 0.49 |
| | DeF | 4.63 | 66.38 | 21.56 | **2.20** | 1.12 | 0.47 | 0.44 | 1.29 |
| | ReC | 4.41 | 66.96 | 20.16 | 2.03 | 1.24 | 0.40 | 0.42 | 1.29 |
| | **CDDFuse** | **4.83** | **88.59** | **33.83** | 2.24 | 1.74 | **0.50** | 0.59 | 1.31 |
| | U2F | 4.88 | 52.98 | 22.54 | 2.08 | 0.75 | 0.37 | 0.46 | 0.49 |
| | SDN | 5.02 | 60.07 | 29.41 | 2.14 | 0.97 | 0.38 | 0.47 | 0.51 |
| | EMF | 4.76 | 72.76 | 22.56 | 2.34 | 1.32 | 0.56 | 0.49 | 1.31 |
| | **CDDFuse\*** | 4.88 | 79.17 | 38.14 | 2.61 | 1.41 | 0.61 | 0.68 | **1.34** |
| **MRI-PET** | **CDDFuse** | 4.24 | 81.72 | 28.04 | 1.87 | 1.82 | 0.66 | 0.65 | 1.46 |
| | **CDDFuse\*** | 4.23 | 70.73 | 29.5 | 2.03 | 1.69 | 0.71 | 0.71 | **1.49** |
| **MRI-SPECT** | **CDDFuse** | 3.91 | 71.82 | 20.68 | 1.89 | 1.92 | 0.66 | 0.69 | 1.44 |
| | **CDDFuse\*** | 3.90 | 58.31 | 20.87 | 2.49 | 1.35 | 0.97 | 0.78 | **1.48** |

**Key takeaways**:
- CDDFuse no fine-tune đã top-2/3 ở mọi dataset → **strong generalization**
- CDDFuse* (fine-tune trên MIF) chỉ tăng SSIM 0.03 → **head-room cho fine-tune nhỏ**
- → Architecture đã sát ceiling; cải tiến phải đến từ **change architecture/loss**, không phải đơn thuần fine-tune

### 12.10 Hiệu chỉnh & lưu ý cho cải tiến

Sau khi đọc paper, tôi điều chỉnh các đề xuất ở §8:

| Đề xuất ở §8 | Hiệu chỉnh |
|---|---|
| **A1 — Hybrid Mamba-Transformer** | Phải biết: paper đã chứng minh **LT > INN cho BTE** và **INN > LT/CNN cho DCE** (ablation Table 2). Nếu thay Mamba, phải compare against LT (cho BTE) và INN (cho DCE) riêng → ablation sẽ phức tạp hơn |
| **A2 — 3-band decomposition** | Decomposition hiện tại được drive bởi L_decomp. Nếu thêm band thứ 3, cần extend L_decomp → cẩn thận với formulation (paper đã thử subtraction vs division, division thắng) |
| **A3 — Cross-modal encoder** | Paper dùng **shared SFE** với lý do "Restormer self-attention across feature dim → cross-modality shallow features without computation overhead". Thay shared bằng 2 encoder riêng có thể mất lợi thế này |
| **B1 — Anatomical loss** | Paper dùng **max-rule** cho L_int^II và L_grad — tức là tin tưởng pixel mạnh hơn ở mỗi vị trí. Anatomical loss có thể conflict với max-rule → cần re-balance |
| **D1 — Color processing** | Paper Figure 6 cho thấy CDDFuse* trên MRI-PET đã preserve color tốt → có thể không phải gap lớn nhất |

### 12.11 Hướng cải tiến mới sau khi đọc paper

Sau khi đọc kỹ paper + ablation, các hướng có novelty cao hơn các đề xuất ban đầu:

#### Hướng N1 — Better Decomposition Loss

Paper cho thấy decomposition loss form ảnh hưởng lớn (Exp I trong ablation). Có thể propose:
- **Mutual Information-based** thay correlation: minimize MI(detail_A, detail_B), maximize MI(base_A, base_B)
- **Wasserstein distance-based** decomposition
- **Information bottleneck** principle

→ Tránh trùng với "vanilla Mamba/Transformer swap".

#### Hướng N2 — Multi-stage > 2-stage

Paper dùng 2-stage. Có thể đề xuất:
- Stage 1: AE training
- Stage 2: Decomposition refinement (only L_decomp)
- Stage 3: Fusion training

→ Chia nhỏ training để mỗi stage focus 1 mục tiêu.

#### Hướng N3 — Replace `max` rule với learnable selector

Paper dùng `max(I_ir, I_vis)` và `max(|∇I_ir|, |∇I_vis|)` làm target — đây là **hard rule**. Đề xuất:
- Learnable per-pixel weighting: w(I_ir, I_vis) thay max
- Spatial attention map drives selection

→ **Concrete contribution**, không bị overlap với existing Mamba paper.

#### Hướng N4 — Test-time MIF specialization (no retrain)

CDDFuse* (fine-tune) chỉ tăng SSIM từ 1.31 → 1.34 (MRI-CT). Có thể:
- Test-time prompt tuning với 1 vài LoRA layer
- Modality embedding injection ở inference time

→ Dùng practical motivation: clinical deployment cần adapt scanner mới mà không retrain toàn model.

---

## 13. Tổng kết

CDDFuse là baseline **rất tốt để extend** vì:
- ✅ Architecture modular, dễ swap component
- ✅ Có 2-stage training stable
- ✅ Top-tier performance ngay khi chưa cải tiến (rank #2)
- ✅ Code clean, well-documented
- ⚠️ Hạn chế: single-scale, 2-band rigid, no medical-aware loss

3 hướng cải tiến đề xuất (Mamba-Hybrid + 3-band + Anatomical loss) có:
- Novelty đủ cho luận văn ThS/TS
- Implementation feasible trong 3 tháng
- Có baseline measurement rõ ràng
- Reviewer-friendly justification

**Bước tiếp theo**: setup môi trường training, reproduce baseline, bắt đầu prototype A1.
