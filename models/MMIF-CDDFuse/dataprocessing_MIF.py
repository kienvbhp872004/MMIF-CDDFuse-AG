"""
Pre-processing cho Medical Image Fusion (paper CDDFuse sect 5.2).

Setup paper-faithful:
- 286 pairs từ Harvard Medical
- Split: 130 train + 20 val + 136 test (21 CT + 42 PET + 73 SPECT)
- Random seed 42 cho reproducibility

Cấu trúc input (đường dẫn relative từ project root):
    ../../Havard-Medical-Image-Fusion-Datasets-main/Havard-Medical-Image-Fusion-Datasets-main/
        CT-MRI/{CT,MRI}/*.png       (184 pairs, L mode 256x256)
        PET-MRI/{PET,MRI}/*.png     (269 pairs, PET=RGB 256x256, MRI=L)
        SPECT-MRI/{SPECT,MRI}/*.png (357 pairs, SPECT=RGB 256x256, MRI=L)

Output:
    data/MIF_train_imgsize_128_stride_64.h5
        groups: mri_patchs, src_patchs (Y channel only, [1,128,128])
    data/MIF_split.json  (lists for train/val/test reproducibility)

Patch extraction: 128x128 với stride 64 → 9 patches/image, low-contrast filtered.

Convention: MRI → "ir" role (anatomical), CT/PET_Y/SPECT_Y → "vi" role (functional).
Khi train: 1 sample = (MRI patch [1,128,128], src patch [1,128,128]).
"""
import json
import os
import random
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------- config
SEED = 42
PATCH_SIZE = 128
STRIDE = 64
LOW_CONTRAST_FRACTION = 0.1

# Test counts paper Tab.5 sect 5.2
N_TEST_CT, N_TEST_PET, N_TEST_SPECT = 21, 42, 73
N_TRAIN, N_VAL = 130, 20

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATASET_ROOT = ROOT / "Havard-Medical-Image-Fusion-Datasets-main" / "Havard-Medical-Image-Fusion-Datasets-main"
OUT_DIR = HERE / "data"
OUT_H5 = OUT_DIR / f"MIF_train_imgsize_{PATCH_SIZE}_stride_{STRIDE}.h5"
OUT_SPLIT = OUT_DIR / "MIF_split.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def rgb2y(img_rgb):
    """[H,W,3] uint8/float -> [H,W] float Y channel."""
    r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def load_pair(src_path, mri_path):
    src = np.array(Image.open(src_path), dtype=np.float32)
    mri = np.array(Image.open(mri_path), dtype=np.float32)
    if src.ndim == 3:
        src = rgb2y(src)
    src = src / 255.0
    mri = mri / 255.0
    return mri, src  # both [H,W] float32 in [0,1]


def extract_patches(img, patch=PATCH_SIZE, stride=STRIDE):
    """[H,W] -> [N, patch, patch]"""
    H, W = img.shape
    ys = list(range(0, H - patch + 1, stride))
    xs = list(range(0, W - patch + 1, stride))
    out = np.zeros((len(ys) * len(xs), patch, patch), dtype=np.float32)
    k = 0
    for y in ys:
        for x in xs:
            out[k] = img[y:y + patch, x:x + patch]
            k += 1
    return out


def is_low_contrast(patch, frac_threshold=LOW_CONTRAST_FRACTION):
    lo, hi = np.percentile(patch, [10, 90])
    if hi < 1e-6:
        return True
    return (hi - lo) / hi < frac_threshold


def collect_pairs(modal_folder, src_subdir):
    """Return list of (src_path, mri_path) tuples."""
    src_dir = DATASET_ROOT / modal_folder / src_subdir
    mri_dir = DATASET_ROOT / modal_folder / "MRI"
    out = []
    for f in sorted(os.listdir(src_dir)):
        if not f.lower().endswith(".png"):
            continue
        mri_path = mri_dir / f
        if mri_path.exists():
            out.append((str(src_dir / f), str(mri_path)))
    return out


def split_286_from_pool():
    """Pick 286 pairs from Harvard pool with paper-faithful test allocation."""
    rng = random.Random(SEED)
    ct = collect_pairs("CT-MRI", "CT")
    pet = collect_pairs("PET-MRI", "PET")
    spect = collect_pairs("SPECT-MRI", "SPECT")
    print(f"[pool] CT={len(ct)} PET={len(pet)} SPECT={len(spect)} total={len(ct)+len(pet)+len(spect)}")

    rng.shuffle(ct)
    rng.shuffle(pet)
    rng.shuffle(spect)

    # Test split per paper
    test_ct = ct[:N_TEST_CT]
    test_pet = pet[:N_TEST_PET]
    test_spect = spect[:N_TEST_SPECT]

    # Remaining pool for train+val (must total 130+20 = 150)
    remain = ct[N_TEST_CT:] + pet[N_TEST_PET:] + spect[N_TEST_SPECT:]
    rng.shuffle(remain)
    pool_286 = remain[: N_TRAIN + N_VAL]  # 150 non-test
    train_pairs = pool_286[:N_TRAIN]
    val_pairs = pool_286[N_TRAIN:N_TRAIN + N_VAL]

    return {
        "train":      [{"src": s, "mri": m} for s, m in train_pairs],
        "val":        [{"src": s, "mri": m} for s, m in val_pairs],
        "test_CT":    [{"src": s, "mri": m} for s, m in test_ct],
        "test_PET":   [{"src": s, "mri": m} for s, m in test_pet],
        "test_SPECT": [{"src": s, "mri": m} for s, m in test_spect],
    }


def main():
    split = split_286_from_pool()
    OUT_SPLIT.write_text(json.dumps(split, indent=2))
    print(f"[split] train={len(split['train'])} val={len(split['val'])} "
          f"test CT/PET/SPECT={len(split['test_CT'])}/{len(split['test_PET'])}/{len(split['test_SPECT'])}")
    print(f"[split] saved -> {OUT_SPLIT}")

    # Patch h5 only for train set
    h5f = h5py.File(OUT_H5, "w")
    g_mri = h5f.create_group("mri_patchs")
    g_src = h5f.create_group("src_patchs")
    train_num = 0
    for pair in tqdm(split["train"], desc="extract patches"):
        mri, src = load_pair(pair["src"], pair["mri"])
        p_mri = extract_patches(mri)
        p_src = extract_patches(src)
        for j in range(p_mri.shape[0]):
            if is_low_contrast(p_mri[j]) or is_low_contrast(p_src[j]):
                continue
            g_mri.create_dataset(str(train_num), data=p_mri[j:j + 1], dtype=p_mri.dtype)
            g_src.create_dataset(str(train_num), data=p_src[j:j + 1], dtype=p_src.dtype)
            train_num += 1
    h5f.close()
    print(f"[h5  ] {train_num} patches saved -> {OUT_H5}")

    # Verify
    with h5py.File(OUT_H5, "r") as f:
        print(f"[h5  ] groups: mri={len(f['mri_patchs'])} src={len(f['src_patchs'])}")


if __name__ == "__main__":
    main()
