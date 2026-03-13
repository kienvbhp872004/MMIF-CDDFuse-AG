import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ==========================================
# ===== CẤU HÌNH - THAY ĐỔI Ở ĐÂY =====
# ==========================================

MAX_IMAGES      = 6          # Số ảnh muốn demo
OUTPUT_FOLDER   = "fusion_demo_output"   # Tên folder lưu ảnh
SAVE_DPI        = 150        # Chất lượng ảnh (150=báo cáo, 300=in ấn)
MODALITY_A      = "SPECT"    # Tên ảnh nguồn A (chỉ để hiển thị)
MODALITY_B      = "MRI"      # Tên ảnh nguồn B (chỉ để hiển thị)

dataset_dir = r"/Havard-Medical-Image-Fusion-Datasets-main"
fused_pair  = "SPECT-MRI"
dir_A       = os.path.join(dataset_dir, fused_pair, "SPECT")
dir_B       = os.path.join(dataset_dir, fused_pair, "MRI")

# ==========================================

from algorithm.wavelet_transform import wavelet_fusion

# Seaborn theme
sns.set_theme(style="dark", font_scale=1.1)
CMAP = "gray"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

images = sorted(os.listdir(dir_A))[:MAX_IMAGES]

print(f"Demo {len(images)} ảnh → folder: '{OUTPUT_FOLDER}'\n")

# ── 1. Lưu từng ảnh riêng: 3 cột (A | B | Fused) ────────────────────────────
saved_figs = []

for idx, img_name in enumerate(images, 1):
    path_A = os.path.join(dir_A, img_name)
    path_B = os.path.join(dir_B, img_name)

    img_A = cv2.imread(path_A, cv2.IMREAD_GRAYSCALE)
    img_B = cv2.imread(path_B, cv2.IMREAD_GRAYSCALE)

    if img_A is None or img_B is None:
        print(f"  [SKIP] Không đọc được: {img_name}")
        continue

    img_fused = wavelet_fusion(img_A, img_B)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    fig.patch.set_facecolor("#1a1a2e")

    panels = [
        (img_A,     MODALITY_A,      "#e94560"),
        (img_B,     MODALITY_B,      "#0f3460"),
        (img_fused, "Fused Result",  "#16213e"),
    ]

    for ax, (data, title, bg) in zip(axes, panels):
        ax.set_facecolor(bg)
        im = ax.imshow(data, cmap=CMAP, interpolation="lanczos")
        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(im.axes, "yticklabels"), color="white")

    stem = os.path.splitext(img_name)[0]
    fig.suptitle(
        f"{MODALITY_A} + {MODALITY_B}  →  Wavelet Fusion   [{stem}]",
        color="white", fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_FOLDER, f"fusion_{stem}.png")
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    saved_figs.append(out_path)
    print(f"  [{idx:02d}] Saved: {out_path}")


# ── 2. Tổng hợp: grid tất cả ảnh vào 1 file ──────────────────────────────────
n = len(saved_figs)
cols = 2          # 2 bộ ảnh (mỗi bộ = 3 cột) trên 1 hàng
rows = (n + cols - 1) // cols

fig_summary, axes_all = plt.subplots(
    rows * 2, cols * 3,
    figsize=(cols * 13, rows * 8.5)
)
fig_summary.patch.set_facecolor("#0d0d0d")

for ax in axes_all.flatten():
    ax.axis("off")
    ax.set_facecolor("#0d0d0d")

for i, img_name in enumerate(images[:n]):
    path_A    = os.path.join(dir_A, img_name)
    path_B    = os.path.join(dir_B, img_name)
    img_A     = cv2.imread(path_A, cv2.IMREAD_GRAYSCALE)
    img_B     = cv2.imread(path_B, cv2.IMREAD_GRAYSCALE)
    img_fused = wavelet_fusion(img_A, img_B)

    row_block = (i // cols) * 2
    col_block = (i  % cols) * 3
    stem      = os.path.splitext(img_name)[0]

    for j, (data, lbl) in enumerate([
        (img_A,     MODALITY_A),
        (img_B,     MODALITY_B),
        (img_fused, "Fused"),
    ]):
        ax = axes_all[row_block, col_block + j]
        ax.imshow(data, cmap=CMAP, interpolation="lanczos")
        ax.set_title(lbl, color="white", fontsize=10, fontweight="bold")
        ax.axis("off")

    # label tên ảnh ở hàng phụ giữa
    label_ax = axes_all[row_block + 1, col_block + 1]
    label_ax.text(
        0.5, 0.5, stem,
        ha="center", va="center", color="#aaaaaa",
        fontsize=9, style="italic", transform=label_ax.transAxes
    )

fig_summary.suptitle(
    f"Wavelet Fusion Summary  ·  {MODALITY_A} + {MODALITY_B}  ·  {n} images",
    color="white", fontsize=16, fontweight="bold", y=1.005
)
plt.tight_layout()

summary_path = os.path.join(OUTPUT_FOLDER, "_summary_all_SPECT_MRI.png")
fig_summary.savefig(summary_path, dpi=SAVE_DPI, bbox_inches="tight",
                    facecolor=fig_summary.get_facecolor())
plt.close(fig_summary)

print(f"\n✅ Summary grid saved: {summary_path}")
print(f"✅ Tổng cộng {len(saved_figs)} ảnh riêng + 1 summary → '{OUTPUT_FOLDER}/'")