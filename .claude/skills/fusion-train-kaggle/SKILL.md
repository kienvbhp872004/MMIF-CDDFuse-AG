---
name: fusion-train-kaggle
description: Train CDDFuse variant trên Kaggle (Tesla P100 với CPU fallback do sm_60 không tương thích torch hiện tại — chấp nhận chậm hơn để KHÔNG phải đợi T4 free quota). Quản lý notebook lifecycle, pull output, đồng bộ vào results_v2/ và PROGRESS.md. Trigger khi user nói "train trên kaggle", "push kaggle", "kéo ckpt về", "kaggle kernel".
---

# fusion-train-kaggle

Mục đích: cho phép user không có GPU mạnh local vẫn chạy được full roadmap. Đẩy training lên Kaggle (free tier 30h GPU/tuần), giữ skill `fusion-bench-variant` + `fusion-stats` chạy local trên ckpt downloaded.

## Khi nào kích hoạt

Trigger:
- "train trên kaggle" / "push kaggle" / "kéo kết quả từ kaggle"
- "không có GPU" / "GPU local yếu"
- User vừa code xong variant, hỏi "train ở đâu giờ"

KHÔNG kích hoạt:
- Variant inference-only (không cần train) — chạy local thẳng
- User đã có GPU local đủ → khuyên local (nhanh hơn, không round-trip upload/download)

## Prerequisites

Skill phải verify đủ trước khi làm bất cứ gì:

1. **`kaggle` CLI installed**: `kaggle --version`. Nếu thiếu: `pip install kaggle`
2. **API key**: `~/.kaggle/kaggle.json` exists, mode 600. Nếu thiếu: hướng dẫn user vào Kaggle Account → Create New Token, save vào `~/.kaggle/kaggle.json`, `chmod 600`
3. **Repo public hoặc đã upload code as Kaggle Dataset**: notebook cần clone code từ đâu đó. Nếu repo private → block và yêu cầu user push code as Kaggle Dataset trước (xem Setup §a).
4. **Pretrained baseline ckpt**: file `models/MMIF-CDDFuse/models/CDDFuse_MIF.pth` cần có sẵn trên Kaggle (làm Dataset 1 lần) — nếu chưa, skill upload trước.

## Cấu trúc folder local

```
kaggle_run/
├── kernel-metadata.json       # config notebook (auto-generated per variant)
├── train_template.ipynb        # notebook template (1 lần, dùng chung)
├── _datasets/
│   ├── harvard-medical/       # synced từ data/reference/
│   └── cddfuse-pretrained/   # synced từ models/MMIF-CDDFuse/models/
└── _runs/
    └── <variant-tag>/         # output từng variant pull về
```

## 3 Sub-actions

### a) Setup (1 lần đầu)

Skill detect chưa setup → chạy:

1. Tạo Kaggle Dataset cho Harvard data:
   ```
   cd kaggle_run/_datasets/harvard-medical/
   kaggle datasets init -p .
   # edit dataset-metadata.json: title="harvard-medical-fusion", id="<user>/harvard-medical-fusion"
   # copy data/reference/CT-MRI/ PET-MRI/ SPECT-MRI/ vào đây
   kaggle datasets create -p .
   ```

2. Tạo Kaggle Dataset cho pretrained ckpt:
   ```
   cd kaggle_run/_datasets/cddfuse-pretrained/
   # copy CDDFuse_MIF.pth + variant ckpts mới (sẽ update sau)
   kaggle datasets create -p .
   ```

3. Verify Kaggle account quota: `kaggle kernels list -m --user <user>` → đủ 30h/tuần.

→ Sau setup này, mỗi lần train variant **không cần re-upload data**.

### b) Train variant

Input: tên variant `CDDFuse-<Module>-<Variant>` (ví dụ `CDDFuse-FuseRule-Gated`) + branch git chứa code variant.

Pre-flight:
- Branch đã commit và push lên `origin`? `git log origin/<branch>..HEAD` empty?
- Notebook template `train_template.ipynb` exists?
- Variant inference script đã hỗ trợ `--save_perimage`?

Generate `kernel-metadata.json`:
```json
{
  "id": "<user>/cddfuse-<module>-<variant>-train",
  "title": "CDDFuse-<Module>-<Variant> training",
  "code_file": "train_<variant>.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "true",
  "dataset_sources": [
    "<user>/harvard-medical-fusion",
    "<user>/cddfuse-pretrained"
  ],
  "competition_sources": [],
  "kernel_sources": []
}
```

Generate notebook `train_<variant>.ipynb` từ template, inject:
- Repo URL + branch + commit sha (pin chính xác để reproducible)
- Variant tag
- Train mode: `light_retrain` (default) hoặc `full_retrain`
- Hyperparams (lấy từ `_ablation_stamp.json` nếu có, hoặc từ paper default)
- Seed = 42

Notebook nội dung core (template):
```python
# Cell 1: clone code
!git clone https://github.com/kienvbhp872004/Image-Fusion.git
%cd Image-Fusion
!git checkout <commit-sha>

# Cell 2: setup
!pip install -q -r requirements.txt
import os, sys, torch, random, numpy as np
torch.manual_seed(42); np.random.seed(42); random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'

# Cell 3: link Kaggle datasets
import shutil
shutil.copytree('/kaggle/input/harvard-medical-fusion', 'data/reference/', dirs_exist_ok=True)
shutil.copy('/kaggle/input/cddfuse-pretrained/CDDFuse_MIF.pth',
            'models/MMIF-CDDFuse/models/')

# Cell 4: train (light retrain — freeze encoder/decoder, train new module)
%cd models/MMIF-CDDFuse
!python train_variant.py \
    --variant <Tag> \
    --pretrained models/CDDFuse_MIF.pth \
    --train_mode light_retrain \
    --epochs 25 \
    --output /kaggle/working/

# Cell 5: inference + per-image CSV (3 modal)
for modal in ['CT', 'PET', 'SPECT']:
    !python evaluate_cddfuse.py \
        --variant <Tag> \
        --modal $modal \
        --ckpt /kaggle/working/CDDFuse-<Tag>.pth \
        --harvard_root ../../data/reference/ \
        --out_dir /kaggle/working/CDDFuse-<Tag>/ \
        --save_perimage

# Cell 6: package output
!cd /kaggle/working && tar czf CDDFuse-<Tag>_results.tar.gz CDDFuse-<Tag>/
```

Push & run:
```
kaggle kernels push -p kaggle_run/
# kernel chạy auto vì có enable_gpu=true
```

In ra cho user:
```
[kaggle] Pushed: <user>/cddfuse-<module>-<variant>-train
[kaggle] Status URL: https://www.kaggle.com/<user>/cddfuse-...
[kaggle] Estimated runtime: 2-3h (light_retrain) / 12-18h (full_retrain)
[kaggle] Will notify when done. Or run: kaggle kernels status <id>
```

### c) Pull results

User: "kéo ckpt về" / "lấy kết quả kaggle"

Steps:
1. `kaggle kernels status <id>` → check complete
2. `kaggle kernels output <id> -p kaggle_run/_runs/<variant-tag>/`
3. Extract: `tar xzf CDDFuse-<Tag>_results.tar.gz`
4. Move:
   - ckpt → `models/MMIF-CDDFuse/models/CDDFuse-<Tag>.pth`
   - results → `results_v2/CDDFuse-<Tag>/` (Fusion/, perimage/, summary CSV)
5. Sinh `_ablation_stamp.json` với extra fields:
   ```json
   "kaggle": {
     "kernel_id": "<user>/cddfuse-<module>-<variant>-train",
     "kernel_version": <int>,
     "runtime_seconds": <int>,
     "gpu_type": "T4" or "P100",
     "kernel_url": "https://..."
   }
   ```
6. Verify integrity:
   - ckpt sha256
   - perimage CSV row count = 24/modal × 3 modal = 72
   - summary CSV exists
7. Hand-off:
   ```
   [kaggle] Pulled: results_v2/CDDFuse-<Tag>/
   [kaggle] Next: run `fusion-stats` to compare with baseline
   ```

## Hard rules

- KHÔNG push code chứa data nhạy cảm (API keys, credentials) lên Kaggle. Skill scan `.env`, `kaggle.json`, `*.pem` trong repo trước khi push.
- KHÔNG để kernel public nếu repo private — `is_private: true` mặc định.
- KHÔNG re-upload Harvard dataset mỗi lần — chỉ upload 1 lần ở Setup, mọi variant share.
- KHÔNG quên pin commit sha trong notebook — branch HEAD có thể move, kết quả không reproduce.
- KHÔNG bỏ qua seed = 42 ở Kaggle (môi trường khác local, seed quan trọng hơn để compare cross-platform).
- LUÔN tar gzip output trước khi pull (Kaggle output dir có nhiều ảnh, tar gọn hơn).

## Light retrain template — chi tiết Phase II

Vì user chưa có `train_variant.py` (chỉ có `evaluate_cddfuse.py`), skill cũng cung cấp template `train_variant.py`:

```python
# models/MMIF-CDDFuse/train_variant.py (skeleton)
"""
Light retrain: freeze SFE/BTE/DCE/Decoder, train only:
  - New module (FusionGate, CrossAttn, etc.) — kiến trúc lấy từ --variant
  - BaseFuseLayer, DetailFuseLayer (fine-tune)

Loss: full L_total^II từ paper §3.5 Eq.10.
Epochs: 25 (default), batch 8, lr 1e-4 cosine decay.
"""
# (Skill sẽ generate full file khi user yêu cầu, dựa trên variant module type)
```

Skill yêu cầu user implement module mới trong `models/MMIF-CDDFuse/net_variants.py` (file mới, không đụng `net.py` baseline).

## Quota management

Trước khi push, in:
```
[kaggle] Quota check: 18.5h / 30h used this week (61%)
[kaggle] This run: ~2.5h estimated
[kaggle] After run: 21h / 30h (70%) — OK
```

Nếu quota < ETA → BLOCK, đề xuất chờ tuần sau hoặc local.

## Failure modes

| Lỗi | Action |
|---|---|
| Kernel timeout 12h | Skill prompt: split thành 2 sessions với resume từ epoch ckpt |
| OOM (T4 chỉ 16GB) | Giảm batch_size từ 8 → 4 trong notebook, retry |
| Internet disabled trên Kaggle | Pre-cache pip packages vào dataset, switch sang offline mode |
| Output tar > 20GB | Loại Fusion images khỏi tar, chỉ giữ ckpt + CSV |

## Hand-off với 3 skill khác

```
fusion-gap-analysis    →  pick variant
fusion-train-kaggle    →  train trên Kaggle GPU      ← skill này
fusion-bench-variant   →  verify output + stamp local
fusion-stats           →  significance test
```

→ Skill này thay thế **bước "train" trong roadmap**, không thay thế các skill khác.
