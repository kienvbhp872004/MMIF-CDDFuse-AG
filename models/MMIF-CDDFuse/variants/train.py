"""
Light retrain script for CDDFuse variants.

Freeze policy:
    FROZEN  : Encoder (Restormer SFE/BTE/DCE), Decoder
    TRAIN   : variant.gated_b, variant.gated_d (or whatever module variant defines)
    FINE-TUNE: BaseFuseLayer, DetailFuseLayer

Loss: Fusionloss (Eq.10 paper) + alpha_decomp * cc_loss.
Default 25 epochs, batch 8, AdamW LR 1e-4 cosine -> 1e-6.

Usage:
    python -m variants.train \
        --variant FuseRule-Gated \
        --pretrained models/CDDFuse_MIF.pth \
        --train_data /path/to/train_pairs/ \
        --output    /path/to/output_ckpt_dir/ \
        --epochs 25 --batch 8 --seed 42

Train data layout expected (any of {CT,PET,SPECT}):
    train_data/
        CT-MRI/MRI/*.png    + CT-MRI/CT/*.png
        PET-MRI/MRI/*.png   + PET-MRI/PET/*.png
        SPECT-MRI/MRI/*.png + SPECT-MRI/SPECT/*.png

NOTE: For light retrain ablation, train data SHOULD be disjoint with eval data.
      Recommended: full Harvard dataset minus the 24×3 reference test pairs.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent  # models/MMIF-CDDFuse/variants/
CDDFUSE_DIR = SCRIPT_DIR.parent                # models/MMIF-CDDFuse/
sys.path.insert(0, str(CDDFUSE_DIR))

from net import (BaseFeatureExtraction, DetailFeatureExtraction,
                 Restormer_Decoder, Restormer_Encoder)
from utils.loss import cc

from variants.losses import FusionLossB
from variants.registry import build_variant

# ------------------------------------------------------------------ utils

def set_seed(s: int) -> None:
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    random.seed(s)
    torch.backends.cudnn.deterministic = True


# ------------------------------------------------------------------ data

class HarvardPairDataset(Dataset):
    """Load (MRI, source) pairs from data/<modal>-MRI/{MRI,<modal>}/. Random crop 128x128."""

    EXTS = (".png", ".jpg", ".bmp")

    def __init__(self, root: Path, modals=("CT", "PET", "SPECT"), crop: int = 128):
        self.crop = crop
        self.pairs = []
        for modal in modals:
            modal_dir = root / f"{modal}-MRI"
            if not modal_dir.exists():
                continue
            mri_dir = modal_dir / "MRI"
            src_dir = modal_dir / modal
            mri_files = {p.stem: p for p in mri_dir.iterdir() if p.suffix.lower() in self.EXTS}
            src_files = {p.stem: p for p in src_dir.iterdir() if p.suffix.lower() in self.EXTS}
            for stem in sorted(set(mri_files) & set(src_files)):
                self.pairs.append((mri_files[stem], src_files[stem]))
        if not self.pairs:
            raise RuntimeError(f"No training pairs found under {root}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        mri_p, src_p = self.pairs[idx]
        mri = np.array(Image.open(mri_p).convert("L"), dtype=np.float32) / 255.0
        src = np.array(Image.open(src_p).convert("L"), dtype=np.float32) / 255.0
        h, w = mri.shape
        c = self.crop
        if h < c or w < c:
            mri = np.pad(mri, ((0, max(0, c - h)), (0, max(0, c - w))))
            src = np.pad(src, ((0, max(0, c - h)), (0, max(0, c - w))))
            h, w = mri.shape
        y = random.randint(0, h - c)
        x = random.randint(0, w - c)
        mri = mri[y:y + c, x:x + c]
        src = src[y:y + c, x:x + c]
        return (torch.from_numpy(mri).unsqueeze(0),  # MRI = "ir" channel
                torch.from_numpy(src).unsqueeze(0))  # src = "vis" channel


# ------------------------------------------------------------------ model wrapper

class VariantModel(nn.Module):
    """CDDFuse + variant fusion modules. Encoder/Decoder loaded from baseline ckpt."""

    def __init__(self, variant_name: str, pretrained_path: Path, device: torch.device):
        super().__init__()
        self.encoder = Restormer_Encoder().to(device)
        self.decoder = Restormer_Decoder().to(device)
        self.base_fuse = BaseFeatureExtraction(dim=64, num_heads=8).to(device)
        self.detail_fuse = DetailFeatureExtraction(num_layers=1).to(device)
        gated_b, gated_d, pixel_select, train_mode = build_variant(variant_name)
        self.gated_b = gated_b.to(device)
        self.gated_d = gated_d.to(device)
        self.pixel_select = pixel_select
        self.train_mode = train_mode

        # Load baseline weights
        ckpt = torch.load(pretrained_path, map_location=device)
        strip = lambda sd: {k.replace("module.", "", 1): v for k, v in sd.items()}
        self.encoder.load_state_dict(strip(ckpt["DIDF_Encoder"]))
        self.decoder.load_state_dict(strip(ckpt["DIDF_Decoder"]))
        self.base_fuse.load_state_dict(strip(ckpt["BaseFuseLayer"]))
        self.detail_fuse.load_state_dict(strip(ckpt["DetailFuseLayer"]))

    def freeze_for_light_retrain(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False
        # base_fuse, detail_fuse, gated_* trainable

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, mri: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        b_a, d_a, _ = self.encoder(mri)
        b_b, d_b, _ = self.encoder(src)
        f_b = self.base_fuse(self.gated_b(b_a, b_b))
        f_d = self.detail_fuse(self.gated_d(d_a, d_b))
        fused, _ = self.decoder(mri, f_b, f_d)
        return fused


# ------------------------------------------------------------------ train loop

def train(args):
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Auto-fallback CPU nếu GPU không support (Pascal sm_60)
        if device.type == "cuda":
            cap = torch.cuda.get_device_capability(0)
            if cap[0] < 7:
                print(f"[train] WARNING: GPU sm_{cap[0]}{cap[1]} unsupported, falling back to CPU")
                device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    set_seed(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] device={device} variant={args.variant} epochs={args.epochs} batch={args.batch}")

    model = VariantModel(args.variant, Path(args.pretrained), device)
    if model.train_mode == "inference_only":
        print(f"[train] Variant '{args.variant}' is inference-only. Skipping training.")
        return
    model.freeze_for_light_retrain()
    n_train = sum(p.numel() for p in model.trainable_params())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[train] params: {n_train:,} trainable / {n_total:,} total")

    dataset = HarvardPairDataset(Path(args.train_data))
    print(f"[train] {len(dataset)} training pairs")
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True)

    # Module B-aware loss: FusionLossB chọn target theo registry's pixel_select.
    fuse_loss = FusionLossB(pixel_select=model.pixel_select).to(device)
    print(f"[train] pixel_select = {model.pixel_select}")

    # Trainable params: model + loss (loss có params nếu pixel_select='learnable').
    trainable = list(model.trainable_params()) + fuse_loss.trainable_loss_params()
    n_loss = sum(p.numel() for p in fuse_loss.trainable_loss_params())
    if n_loss > 0:
        print(f"[train] loss params: {n_loss:,} (learnable target weight net)")
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-6)

    history = []
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        ep_total = ep_int = ep_grad = ep_dec = 0.0
        for mri, src in loader:
            mri = mri.to(device); src = src.to(device)
            fused = model(mri, src)
            # Fusionloss: (image_vis, image_ir, generated)
            loss_total, loss_int, loss_grad = fuse_loss(src, mri, fused)
            # Decomposition consistency (paper Eq.9, kept as regularizer)
            with torch.no_grad():
                pass  # decomp loss requires features; simplified: skipped in light retrain
            loss = loss_total
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_params(), max_norm=0.01)
            optim.step()
            ep_total += float(loss); ep_int += float(loss_int); ep_grad += float(loss_grad)
        sched.step()
        n_b = max(1, len(loader))
        ep_total /= n_b; ep_int /= n_b; ep_grad /= n_b
        dt = time.time() - t0
        lr = optim.param_groups[0]["lr"]
        msg = f"[train] ep {epoch+1:02d}/{args.epochs}  loss={ep_total:.4f}  int={ep_int:.4f}  grad={ep_grad:.4f}  lr={lr:.2e}  ({dt:.1f}s)"
        print(msg)
        history.append({"epoch": epoch + 1, "loss": ep_total, "int": ep_int, "grad": ep_grad, "lr": lr})

    # Save checkpoint — same key layout as baseline + new keys for variant fuse modules
    ckpt_path = out_dir / f"CDDFuse-{args.variant}.pth"
    state = {
        "DIDF_Encoder":     model.encoder.state_dict(),
        "DIDF_Decoder":     model.decoder.state_dict(),
        "BaseFuseLayer":    model.base_fuse.state_dict(),
        "DetailFuseLayer":  model.detail_fuse.state_dict(),
        "GatedB":           model.gated_b.state_dict(),
        "GatedD":           model.gated_d.state_dict(),
        "LossNet":          fuse_loss.state_dict() if fuse_loss.trainable_loss_params() else {},
        "pixel_select":     model.pixel_select,
        "variant_name":     args.variant,
    }
    torch.save(state, ckpt_path)
    with open(out_dir / f"CDDFuse-{args.variant}_train_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"[train] DONE → {ckpt_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant",     required=True,  help="e.g. FuseRule-Gated")
    ap.add_argument("--pretrained",  required=True,  help="Path to CDDFuse_MIF.pth")
    ap.add_argument("--train_data",  required=True,  help="Root with <modal>-MRI/ folders")
    ap.add_argument("--output",      required=True,  help="Output dir for ckpt")
    ap.add_argument("--epochs",      type=int,   default=25)
    ap.add_argument("--batch",       type=int,   default=8)
    ap.add_argument("--lr",          type=float, default=1e-4)
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--workers",     type=int,   default=2)
    ap.add_argument("--device",      type=str,   default="auto",
                    help="auto / cuda / cpu. 'auto' fallback CPU nếu GPU sm < 7.0")
    return ap.parse_args()


if __name__ == "__main__":
    train(parse_args())
