"""
Variant registry: ánh xạ tên variant -> (BaseFuseModule, DetailFuseModule, pixel_select, train_mode).

Module A — Fusion Rule: thay phép `+` trong BaseFuseLayer/DetailFuseLayer.
                        pixel_select = 'max' (paper baseline cho loss).
Module B — PixelSelect: giữ FuseRule = Sum (IdentitySum), thay rule trong L_int^II loss.
                        BaseFuseLayer/DetailFuseLayer fine-tune với target khác.

Khi thêm variant mới:
  1. Thêm class vào modules.py (Module A) hoặc strategy vào losses.py (Module B)
  2. Thêm 1 dòng vào VARIANT_REGISTRY ở đây
  3. Không đụng nơi khác
"""
from typing import Callable, Dict, Tuple

import torch.nn as nn

from .modules import ChannelMoEFuse, CrossAttnFuse, GatedFuseLayer, IdentitySum

# (base_factory, detail_factory, pixel_select, train_mode)
# pixel_select ∈ {"max", "mean", "saliency", "learnable"}
# train_mode    ∈ {"light_retrain", "full_retrain", "inference_only"}
VARIANT_REGISTRY: Dict[str, Tuple[Callable[[], nn.Module], Callable[[], nn.Module], str, str]] = {
    # Module A — Fusion Rule (loss = max baseline)
    "FuseRule-Sum":        (lambda: IdentitySum(),              lambda: IdentitySum(),              "max", "inference_only"),
    "FuseRule-Gated":      (lambda: GatedFuseLayer(64),         lambda: GatedFuseLayer(64),         "max", "light_retrain"),
    "FuseRule-CrossAttn":  (lambda: CrossAttnFuse(64, 4),       lambda: CrossAttnFuse(64, 4),       "max", "light_retrain"),
    "FuseRule-ChannelMoE": (lambda: ChannelMoEFuse(64, 16),     lambda: ChannelMoEFuse(64, 16),     "max", "light_retrain"),

    # Module B — PixelSelect (FuseRule = Sum, only loss target differs)
    "PixelSelect-Max":       (lambda: IdentitySum(),            lambda: IdentitySum(),            "max",       "inference_only"),
    "PixelSelect-Mean":      (lambda: IdentitySum(),            lambda: IdentitySum(),            "mean",      "light_retrain"),
    "PixelSelect-Saliency":  (lambda: IdentitySum(),            lambda: IdentitySum(),            "saliency",  "light_retrain"),
    "PixelSelect-Learnable": (lambda: IdentitySum(),            lambda: IdentitySum(),            "learnable", "light_retrain"),

    # Combined — Module A winner (Gated) × Module B winner (Saliency)
    "Combined-Gated-Saliency":  (lambda: GatedFuseLayer(64),      lambda: GatedFuseLayer(64),      "saliency",  "light_retrain"),
    "Combined-Gated-Learnable":    (lambda: GatedFuseLayer(64),   lambda: GatedFuseLayer(64),   "learnable", "light_retrain"),
    "Combined-Gated-Saliency-CKA": (lambda: GatedFuseLayer(64),   lambda: GatedFuseLayer(64),   "saliency",  "light_retrain"),  # Module C
}


def build_variant(name: str) -> Tuple[nn.Module, nn.Module, str, str]:
    """
    Lấy (base_module, detail_module, pixel_select, train_mode) cho variant.

    name: ví dụ "FuseRule-Gated", "PixelSelect-Mean".
          Auto-strip prefix "CDDFuse-" nếu có.
    """
    if name.startswith("CDDFuse-"):
        name = name[len("CDDFuse-"):]
    if name not in VARIANT_REGISTRY:
        raise KeyError(f"Unknown variant '{name}'. Available: {list(VARIANT_REGISTRY)}")
    base_fac, detail_fac, pixel_select, train_mode = VARIANT_REGISTRY[name]
    return base_fac(), detail_fac(), pixel_select, train_mode


def list_variants() -> Dict[str, str]:
    """Trả {name: train_mode} cho debug / CLI listing."""
    return {n: tm for n, (_, _, _, tm) in VARIANT_REGISTRY.items()}
