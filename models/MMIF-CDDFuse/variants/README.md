# CDDFuse Variants

Folder này chứa **toàn bộ code cải tiến** của bạn so với baseline CDDFuse (paper CVPR 2023). `net.py` ở thư mục cha **không bị đụng** — đó là kiến trúc paper gốc, để reviewer dễ verify.

## Cấu trúc

```
variants/
├── README.md         # file này
├── __init__.py
├── modules.py        # mọi class variant (GatedFuseLayer, ...)
├── registry.py       # ánh xạ tên variant → modules
└── train.py          # light retrain script generic
```

## Variants hiện có

| Tên | Module thay thế | Cơ chế | Tham số mới | Train mode |
|---|---|---|---|---|
| `FuseRule-Sum` | (baseline) | `A + B` | 0 | inference_only |
| `FuseRule-Gated` | Fusion rule | `g·A + (1-g)·B`, gate học | ~16K | light_retrain (~2h) |

## Thêm variant mới — 3 bước

### 1. Thêm class trong `modules.py`

```python
class CrossAttnFuse(nn.Module):
    def __init__(self, dim=64, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        ...
    def forward(self, feat_a, feat_b):
        # B,C,H,W → B,(HW),C cho attention
        ...
        return fused
```

Convention: `forward(feat_a, feat_b) -> tensor` cùng shape `feat_a`.

### 2. Đăng ký trong `registry.py`

```python
VARIANT_REGISTRY = {
    ...
    "FuseRule-CrossAttn": (lambda: CrossAttnFuse(64), lambda: CrossAttnFuse(64), "light_retrain"),
}
```

### 3. Cập nhật `train.py` save/load nếu module có state đặc biệt

`train.py` hiện save 2 key: `GatedB`, `GatedD`. Nếu module mới có shape/keys giống, không cần đổi. Nếu khác (ví dụ MoE có expert weights riêng), generalise key naming.

`evaluate_cddfuse.py` cũng cần cập nhật `load_weights()` để load đúng key — hoặc tốt hơn: refactor sang generic key như `VariantBase`/`VariantDetail` (tương lai).

## Naming convention

Variant tag: **`<Module>-<Alternative>`**, PascalCase.

- `Module` ∈ {FuseRule, PixelSelect, DecompLoss, ...}
- `Alternative` mô tả cụ thể: `Sum`, `Gated`, `CrossAttn`, `ChannelMoE`...

Đặt tên đúng convention vì:
- Skill `fusion-stats` group variants theo prefix `Module-` cho Friedman test.
- Output dir trong `results_v2/CDDFuse-FuseRule-Gated/` nói rõ "ablation thuộc Module A".

## Train workflow

```bash
# Local hoặc Kaggle
python -m variants.train \
    --variant     FuseRule-Gated \
    --pretrained  models/CDDFuse_MIF.pth \
    --train_data  /path/to/train/ \
    --output      /path/to/output/ \
    --epochs      25 --batch 8 --seed 42
```

Output: `{output}/CDDFuse-FuseRule-Gated.pth` + history JSON.

## Eval với variant

```bash
python evaluate_cddfuse.py \
    --variant     FuseRule-Gated \
    --modal       CT \
    --ckpt        models/CDDFuse-FuseRule-Gated.pth \
    --harvard_root data/reference/ \
    --out_dir     ../../results_v2/CDDFuse-FuseRule-Gated/ \
    --save_perimage
```

`--variant` omit → baseline CDDFuse (sum).
`--save_perimage` REQUIRED nếu sau muốn dùng skill `fusion-stats`.

## Freeze policy của light retrain

```
FROZEN  : Restormer_Encoder, Restormer_Decoder
TRAINED : variant module (gated_b, gated_d), BaseFuseLayer, DetailFuseLayer
```

Lý do: Encoder paper đã pretrain ổn định trên 286 medical pairs. Light retrain chỉ tinh chỉnh fusion stage → đủ thay đổi behavior, không phá representation đã học.

## Cảnh báo data

`train.py` đọc data theo cấu trúc `<modal>-MRI/{MRI,<modal>}/`. **Train data phải DISJOINT với test data**. Nếu bạn dùng `data/reference/` (test) làm train → data leak, kết quả vô nghĩa cho luận văn.

Recommendation: upload riêng dataset `harvard-medical-train` (130 train pairs từ paper split) lên Kaggle, đặt `--train_data` về dataset đó. Eval vẫn dùng `data/reference/` (24+24+24 test).
