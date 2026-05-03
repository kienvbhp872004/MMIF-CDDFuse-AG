"""
Module B — PixelSelect alternatives for L_int^II target.

Paper Eq.10 dùng `target = max(I_y, I_ir)` làm target intensity. Module B test 4 alternatives:
  - max      : paper baseline (hard pixel-wise selection)
  - mean     : (I_y + I_ir) / 2
  - saliency : weight theo |∇I| magnitude (deterministic, no learnable params)
  - learnable: weight map từ small CNN (~80 params)

Gradient loss (L_grad) giữ NGUYÊN max-rule paper cho cả 4 alternatives → ablation chỉ thay đổi
intensity rule, không confound với gradient rule.

Convention: forward(image_vis, image_ir, generate_img) -> (loss_total, loss_int, loss_grad)
matching paper's Fusionloss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Sobelxy(nn.Module):
    """Device-agnostic Sobel via register_buffer (auto-move with .to(device))."""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("weightx", kx)
        self.register_buffer("weighty", ky)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sx = F.conv2d(x, self.weightx, padding=1)
        sy = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sx) + torch.abs(sy)


class FusionLossB(nn.Module):
    """
    Module B fusion loss với 4 PixelSelect strategies cho L_int^II target.

    pixel_select ∈ {'max', 'mean', 'saliency', 'learnable'}
    """

    def __init__(self, pixel_select: str = "max"):
        super().__init__()
        if pixel_select not in {"max", "mean", "saliency", "learnable"}:
            raise ValueError(f"Unknown pixel_select: {pixel_select}")
        self.pixel_select = pixel_select
        self.sobel = Sobelxy()

        if pixel_select == "learnable":
            # Tiny CNN: input (I_y, I_ir, |∇I_y|, |∇I_ir|) → 1 weight map.
            # Init bias=0 weight=0 → sigmoid(0) = 0.5 → epoch-0 = mean target.
            self.weight_net = nn.Sequential(
                nn.Conv2d(4, 8, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(8, 1, kernel_size=1),
            )
            nn.init.zeros_(self.weight_net[-1].weight)
            nn.init.zeros_(self.weight_net[-1].bias)

    def compute_target(self, image_y: torch.Tensor, image_ir: torch.Tensor) -> torch.Tensor:
        if self.pixel_select == "max":
            return torch.max(image_y, image_ir)
        if self.pixel_select == "mean":
            return 0.5 * (image_y + image_ir)
        if self.pixel_select == "saliency":
            g_y = self.sobel(image_y) + 1e-6
            g_ir = self.sobel(image_ir) + 1e-6
            w = g_y / (g_y + g_ir)
            return w * image_y + (1.0 - w) * image_ir
        # learnable
        g_y = self.sobel(image_y)
        g_ir = self.sobel(image_ir)
        feat = torch.cat([image_y, image_ir, g_y, g_ir], dim=1)
        w = torch.sigmoid(self.weight_net(feat))
        return w * image_y + (1.0 - w) * image_ir

    def forward(self, image_vis: torch.Tensor, image_ir: torch.Tensor, generate_img: torch.Tensor):
        # image_vis = source (CT/PET/SPECT), image_ir = MRI per CDDFuse convention.
        image_y = image_vis[:, :1, :, :]

        # L_int^II: intensity loss with strategy-dependent target
        target_in = self.compute_target(image_y, image_ir)
        loss_in = F.l1_loss(target_in, generate_img)

        # L_grad: gradient consistency with max-rule (paper-baseline, fixed across alternatives)
        gy = self.sobel(image_y)
        gir = self.sobel(image_ir)
        gen_g = self.sobel(generate_img)
        joint_g = torch.max(gy, gir)
        loss_grad = F.l1_loss(joint_g, gen_g)

        loss_total = loss_in + 10.0 * loss_grad
        return loss_total, loss_in, loss_grad

    def trainable_loss_params(self):
        """Return list params trong loss (cho learnable). Trống cho max/mean/saliency."""
        if self.pixel_select == "learnable":
            return list(self.weight_net.parameters())
        return []
