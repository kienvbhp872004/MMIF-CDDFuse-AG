"""
Variant modules for CDDFuse fusion-rule ablation (Module A).

Mỗi class ở đây thay thế phép `+` trong:
    f_f_b = base_fuse(f_v_b + f_i_b)
    f_f_d = detail_fuse(f_v_d + f_i_d)

Convention: forward(feat_a, feat_b) -> tensor cùng shape feat_a.
Init phải đảm bảo behavior epoch-0 ≈ baseline (sum hoặc avg) để light retrain hội tụ nhanh.
"""
import torch
import torch.nn as nn


class IdentitySum(nn.Module):
    """Baseline: behavior gốc của paper, fuse = A + B. Không có tham số."""

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        return feat_a + feat_b


class GatedFuseLayer(nn.Module):
    """
    Soft per-channel × per-spatial gate thay cho A + B.

        g     = sigmoid(Conv1x1([A, B]))    # shape [B, dim, H, W]
        out   = g * A + (1 - g) * B

    Init: Conv weight và bias = 0  =>  g = sigmoid(0) = 0.5
          => epoch-0: out = (A + B) / 2  ≈  scaled baseline
    Tham số: dim*2*dim*1*1 + dim ≈ 8K cho dim=64.
    """

    def __init__(self, dim: int = 64):
        super().__init__()
        self.gate = nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([feat_a, feat_b], dim=1)))
        return g * feat_a + (1.0 - g) * feat_b
