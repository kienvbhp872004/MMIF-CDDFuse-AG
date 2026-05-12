"""
Batch runner: chạy các model fusion với data/reference/, output vào results_v2/.

Usage:
    python run_all_v2.py              # chạy hết
    python run_all_v2.py AWFusion     # chỉ 1 model
    python run_all_v2.py --dry-run    # in lệnh, không chạy
"""
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA = REPO / "data" / "reference"
OUT  = REPO / "results_v2"
LOG  = REPO / "results_v2" / "_runner.log"
OUT.mkdir(parents=True, exist_ok=True)

# (folder_in_data/reference, second-source-subfolder-name)
MODAL_PATHS = {
    "CT":    ("CT-MRI", "CT"),
    "PET":   ("PET-MRI", "PET"),
    "SPECT": ("SPECT-MRI", "SPECT"),
}

ALL = ["CT", "PET", "SPECT"]

# Per-model config: (model_dir, out_name, entry, arg_style, modalities)
# arg_style:
#   "mri_pet"   -> --harvard_mri <mri> --harvard_pet <src>
#   "mri_src"   -> --harvard_mri <mri> --harvard_src <src>
#   "root"      -> --harvard_root <data/reference>
#   "dataset"   -> --dataset_root <data/reference>
#   "ecin"      -> --dataset_root <data/reference>  (same as dataset)
#   "hardcoded" -> no args, paths hardcoded in script (need special handling)
MODELS = [
    ("AWFusion",            "AWFusion",     "evaluate_awfusion.py",     "mri_pet",   ALL),
    ("BSAFusion",           "BSAFusion",    "evaluate_bsafusion.py",    "mri_pet",   ALL),
    ("C2RF",                "C2RF",         "evaluate_c2rf.py",         "mri_pet",   ["PET"]),
    ("MMIF-CDDFuse",        "CDDFuse",      "evaluate_cddfuse.py",      "root",      ALL),
    ("CM-CSAMFNet",         "CM-CSAMFNet",  "evaluate_cmcsamfnet.py",   "root",      ALL),
    ("CMMDL",               "CMMDL",        "evaluate_cmmdl.py",        "root",      ALL),
    ("DDBFusion",           "DDBFusion",    "evaluate_ddbfusion.py",    "root",      ALL),
    ("MMIF-DDFM",           "DDFM",         "evaluate_ddfm.py",         "root",      ALL),
    ("DM-FNet",             "DM-FNet",      "evaluate_dm_fnet.py",      "root",      ALL),
    ("ECINFusion",          "ECINFusion",   "evaluate_ecinfusion.py",   "dataset",   ["CT", "PET"]),
    ("GeSeNet",             "GeSeNet",      "evaluate_gesenet.py",      "root",      ALL),
    ("ITFuse",              "ITFuse",       "evaluate_itfuse.py",       "root",      ALL),
    ("LRFNet",              "LRFNet",       "evaluate_lrfnet.py",       "root",      ALL),
    ("MBHFuse",             "MBHFuse",      "evaluate_mbhfuse.py",      "dataset",   ALL),
    ("MFS-Fusion",          "MFS-Fusion",   "evaluate_mfs_fusion.py",   "dataset",   ALL),
    ("imagefusion-nestfuse","NestFuse",     "evaluate_nestfuse.py",     "root",      ALL),
    ("SPDFusion",           "SPDFusion",    "evaluate_spdfusion.py",    "mri_src",   ALL),
    ("TUFusion",            "TUFusion",     "evaluate_tufusion.py",     "mri_src",   ALL),
    ("VDMUFusion",          "VDMUFusion",   "evaluate_vdmufusion.py",   "mri_src",   ALL),
    ("WaveFusion",          "WaveFusion",   "evaluate_wavefusion.py",   "mri_src",   ALL),
    # Hardcoded-path models — patched to read HARVARD_ROOT + OUT_DIR env vars.
    # These run all 3 modalities internally via __main__ loop (ignore per-modality list).
    ("MM-Net-Fusion",       "MM-Net-Fusion","evaluate_mmnet_fusion.py", "envvar",    ALL),
    ("AdaFuse",             "AdaFuse",      "evaluate_adafuse.py",      "envvar",    ALL),
    ("MMIF-INet",           "MMIF-INet",    "evaluate_mmif_inet.py",    "envvar",    ALL),
    ("MLFuse",              "MLFuse",       "evaluate_mlfuse.py",       "envvar",    ALL),
    ("MMAE",                "MMAE",         "evaluate_mmae.py",         "envvar",    ALL),
    ("PSFusion",            "PSFusion",     "evaluate_psfusion.py",     "envvar",    ALL),
    ("SFMFusion",           "SFMFusion",    "evaluate_sfmfusion.py",    "envvar",    ALL),
]


def build_cmd(model_dir, out_name, entry, style, modal):
    pair_dir, src_sub = MODAL_PATHS[modal]
    mri = DATA / pair_dir / "MRI"
    src = DATA / pair_dir / src_sub
    out = OUT / out_name
    entry_path = REPO / "models" / model_dir / entry
    if style == "envvar":
        # Script reads HARVARD_ROOT + OUT_DIR env vars; runs all modalities internally
        return [sys.executable, str(entry_path)]
    cmd = [sys.executable, str(entry_path), "--modal", modal, "--out_dir", str(out)]
    if style == "mri_pet":
        cmd += ["--harvard_mri", str(mri), "--harvard_pet", str(src)]
    elif style == "mri_src":
        cmd += ["--harvard_mri", str(mri), "--harvard_src", str(src)]
    elif style == "root":
        cmd += ["--harvard_root", str(DATA)]
    elif style == "dataset":
        cmd += ["--dataset_root", str(DATA)]
    else:
        return None
    return cmd


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_one(cfg, dry_run=False):
    import os
    model_dir, out_name, entry, style, modalities = cfg
    # envvar style: one invocation runs all modalities internally
    targets = [None] if style == "envvar" else modalities
    for modal in targets:
        label = f"{out_name}/{modal if modal else 'ALL'}"
        cmd = build_cmd(model_dir, out_name, entry, style, modal or modalities[0])
        if cmd is None:
            log(f"SKIP {label}: unsupported style {style}")
            continue
        env = os.environ.copy()
        if style == "envvar":
            env["HARVARD_ROOT"] = str(DATA)
            env["OUT_DIR"] = str(OUT / out_name)
            (OUT / out_name).mkdir(parents=True, exist_ok=True)
        log(f"START {label}")
        if dry_run:
            log("  " + " ".join(cmd))
            if style == "envvar":
                log(f"  env HARVARD_ROOT={env['HARVARD_ROOT']} OUT_DIR={env['OUT_DIR']}")
            continue
        t0 = time.time()
        try:
            p = subprocess.run(cmd, cwd=REPO / "models" / model_dir,
                               capture_output=True, text=True, timeout=14400, env=env)
            dt = time.time() - t0
            if p.returncode == 0:
                log(f"OK    {label} ({dt:.1f}s)")
            else:
                log(f"FAIL  {label} ({dt:.1f}s) rc={p.returncode}")
                tail = (p.stderr or p.stdout or "").strip().splitlines()[-8:]
                for ln in tail:
                    log(f"    | {ln}")
        except subprocess.TimeoutExpired:
            log(f"TIMEOUT {label}")
        except Exception as e:
            log(f"ERROR {label}: {e}")


def main():
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    targets = [a for a in args if not a.startswith("--")]
    to_run = [c for c in MODELS if not targets or c[1] in targets or c[0] in targets]
    log(f"=== Runner start · {len(to_run)} model(s) · dry_run={dry_run} ===")
    for cfg in to_run:
        run_one(cfg, dry_run=dry_run)
    log("=== Runner done ===")


if __name__ == "__main__":
    main()
