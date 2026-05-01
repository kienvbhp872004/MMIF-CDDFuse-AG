"""
Tách train split từ Harvard full dataset, loại trừ test pairs trong data/reference/.
Stage vào kaggle_run/_datasets/harvard-medical-train/ để push lên Kaggle.

Usage:
    python kaggle_run/prepare_train_data.py
"""
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FULL = REPO / "Havard-Medical-Image-Fusion-Datasets-main" / "Havard-Medical-Image-Fusion-Datasets-main"
TEST = REPO / "data" / "reference"
OUT  = REPO / "kaggle_run" / "_datasets" / "harvard-medical-train"

EXTS = (".png", ".jpg", ".bmp")
MODALS = ["CT-MRI", "PET-MRI", "SPECT-MRI"]


def stems(folder: Path) -> set:
    return {p.stem for p in folder.iterdir() if p.suffix.lower() in EXTS}


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    summary = []
    for modal in MODALS:
        modal_short = modal.replace("-MRI", "")
        full_mri = FULL / modal / "MRI"
        full_src = FULL / modal / modal_short
        test_mri = TEST / modal / "MRI"

        test_stems = stems(test_mri) if test_mri.exists() else set()
        full_stems = stems(full_mri)
        train_stems = full_stems - test_stems

        out_mri = OUT / modal / "MRI"
        out_src = OUT / modal / modal_short
        out_mri.mkdir(parents=True)
        out_src.mkdir(parents=True)

        copied = 0
        for stem in sorted(train_stems):
            for ext in EXTS:
                m = full_mri / f"{stem}{ext}"
                s = full_src / f"{stem}{ext}"
                if m.exists() and s.exists():
                    shutil.copy(m, out_mri / m.name)
                    shutil.copy(s, out_src / s.name)
                    copied += 1
                    break
        summary.append((modal, len(full_stems), len(test_stems), copied))
        print(f"  {modal:12s}  full={len(full_stems):3d}  test={len(test_stems):2d}  train={copied:3d}")

    total = sum(c for _, _, _, c in summary)
    print(f"\nTotal train pairs staged: {total}")
    print(f"Output: {OUT}")

    # Kaggle metadata
    meta = OUT / "dataset-metadata.json"
    meta.write_text(
        '{\n'
        '  "title": "Harvard Medical Image Fusion - Train Split",\n'
        '  "id": "kienvbhp1234/harvard-medical-train",\n'
        '  "licenses": [{"name": "CC0-1.0"}]\n'
        '}\n'
    )
    print(f"Metadata: {meta}")
    print(f"\nNext: kaggle datasets create -p {OUT} --dir-mode zip")


if __name__ == "__main__":
    main()
