# Kaggle training workspace

Folder dành cho skill `fusion-train-kaggle`. Không commit folder `_runs/` và `_datasets/` (đã trong .gitignore).

## Cấu trúc

```
kaggle_run/
├── README.md                  # file này
├── train_template.ipynb        # template notebook (skill auto-generate per variant)
├── kernel-metadata.json        # generated per-variant
├── _datasets/                  # (gitignored) local copy data trước khi push lên Kaggle
└── _runs/                      # (gitignored) output pull về từ Kaggle kernel
```

## Setup 1 lần

1. Cài kaggle CLI:
   ```
   pip install kaggle
   ```

2. Lấy API token: vào https://www.kaggle.com/settings → "Create New Token" → tải `kaggle.json`

3. Đặt vào:
   ```
   # Windows
   C:\Users\<user>\.kaggle\kaggle.json
   # Linux/Mac
   ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

4. Verify:
   ```
   kaggle datasets list -m
   ```

## Sau setup

Gọi skill `fusion-train-kaggle` từ Claude Code:
- "train CDDFuse-FuseRule-Gated trên kaggle"
- "kéo kết quả CDDFuse-FuseRule-Gated về"

Skill sẽ tự handle phần còn lại.

## Quota tracking

Free tier Kaggle:
- **30h GPU/tuần** (T4 hoặc P100)
- **12h max/session** (cần resume nếu vượt)
- **20GB output/run**
- Reset reset thứ Bảy 00:00 UTC

Check còn lại:
```
kaggle kernels list -m --user <your-username>
```
