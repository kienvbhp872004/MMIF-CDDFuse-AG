"""
Variant registry: ánh xạ tên variant -> (BaseFuseModule, DetailFuseModule, train_mode).

Khi thêm alternative mới (CrossAttn, ChannelMoE, ...):
  1. Thêm class vào modules.py
  2. Thêm 1 dòng vào VARIANT_REGISTRY ở đây
  3. Không đụng nơi khác
"""
from typing import Callable, Dict, Optional, Tuple

import torch.nn as nn

from .modules import CrossAttnFuse, GatedFuseLayer, IdentitySum

# (base_factory, detail_factory, train_mode)
# train_mode in {"light_retrain", "full_retrain", "inference_only"}
VARIANT_REGISTRY: Dict[str, Tuple[Callable[[], nn.Module], Callable[[], nn.Module], str]] = {
    # Baseline — paper gốc
    "FuseRule-Sum":     (lambda: IdentitySum(),         lambda: IdentitySum(),         "inference_only"),
    # Module A alternatives
    "FuseRule-Gated":     (lambda: GatedFuseLayer(64),               lambda: GatedFuseLayer(64),               "light_retrain"),
    "FuseRule-CrossAttn": (lambda: CrossAttnFuse(64, num_heads=4),   lambda: CrossAttnFuse(64, num_heads=4),   "light_retrain"),
    # TODO: FuseRule-ChannelMoE
}


def build_variant(name: str) -> Tuple[nn.Module, nn.Module, str]:
    """
    Lấy (base_module, detail_module, train_mode) cho variant.

    name: tên variant không có prefix "CDDFuse-", ví dụ "FuseRule-Gated".
          Nếu user truyền "CDDFuse-FuseRule-Gated" thì auto-strip.
    """
    if name.startswith("CDDFuse-"):
        name = name[len("CDDFuse-"):]
    if name not in VARIANT_REGISTRY:
        raise KeyError(f"Unknown variant '{name}'. Available: {list(VARIANT_REGISTRY)}")
    base_fac, detail_fac, train_mode = VARIANT_REGISTRY[name]
    return base_fac(), detail_fac(), train_mode


def list_variants() -> Dict[str, str]:
    """Trả {name: train_mode} cho debug / CLI listing."""
    return {n: tm for n, (_, _, tm) in VARIANT_REGISTRY.items()}
