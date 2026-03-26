"""
Gaussian / Laplacian Pyramid Image Fusion
==========================================

Thuật toán Pyramid cho image fusion:
- Gaussian Pyramid: giảm dần resolution bằng lowpass + downsample
- Laplacian Pyramid: bandpass = sai khác giữa 2 level liên tiếp Gaussian
- Fusion: merge từng level Laplacian rồi reconstruct

Pipeline:
    Input Image
    → Gaussian Pyramid: [G0, G1, ..., GN]
    → Laplacian Pyramid: [L0, L1, ..., L(N-1), GN]
       Lk = Gk - expand(G(k+1))
    → Fusion rule tại mỗi level
    → Inverse pyramid → Fused Image

References:
    - Burt, P. J., & Adelson, E. H. (1983). The Laplacian Pyramid as a Compact
      Image Code. IEEE Transactions on Communications.
    - Toet, A. (1989). Image fusion by a ratio of low-pass pyramid. Pattern
      Recognition Letters.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

from ._base import BaseFusion


class PyramidFusion(BaseFusion):
    """
    Gaussian / Laplacian Pyramid Fusion.

    Parameters
    ----------
    levels : int
        Số level pyramid (default=5).
    fusion_mode : str
        Chế độ fusion cho Laplacian (bandpass) levels:
        - 'max_abs'    : chọn pixel có |value| lớn hơn (mặc định)
        - 'average'    : trung bình
        - 'weighted'   : weighted average dựa trên local salience
    low_fusion : str
        Chế độ fusion cho top level (low-frequency residual):
        - 'average'    : trung bình (mặc định)
        - 'max'        : chọn pixel lớn hơn
    kernel_size : int
        Kích thước Gaussian kernel cho xây pyramid (default=5).

    Example
    -------
    >>> from algorithm.decomposition import PyramidFusion
    >>> fuser = PyramidFusion(levels=5, fusion_mode='max_abs')
    >>> fused_gray = fuser.fuse(img1_gray, img2_gray)       # grayscale
    >>> fused_color = fuser.fuse_ycbcr(img1_bgr, img2_bgr)  # YCbCr color
    >>> fused_auto = fuser.fuse_auto(img1, img2)             # auto-detect
    """

    def __init__(
        self,
        levels: int = 5,
        fusion_mode: str = 'max_abs',
        low_fusion: str = 'average',
        kernel_size: int = 5,
    ):
        self.levels = levels
        self.fusion_mode = fusion_mode
        self.low_fusion = low_fusion
        self.kernel_size = kernel_size

    # ══════════════════════════════════════════════════════════════════════════
    #  Gaussian Pyramid
    # ══════════════════════════════════════════════════════════════════════════

    def gaussian_pyramid(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Xây Gaussian Pyramid.

        Parameters
        ----------
        img : np.ndarray
            Ảnh input (H, W), float64.

        Returns
        -------
        gp : list[np.ndarray]
            [G0 (original), G1 (1/2 size), ..., GN (1/2^N size)]
        """
        gp = [img.astype(np.float64)]
        current = gp[0]
        for _ in range(self.levels):
            down = cv2.pyrDown(current)
            gp.append(down)
            current = down
        return gp

    # ══════════════════════════════════════════════════════════════════════════
    #  Laplacian Pyramid
    # ══════════════════════════════════════════════════════════════════════════

    def laplacian_pyramid(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Xây Laplacian Pyramid.

        Returns
        -------
        lp : list[np.ndarray]
            [L0, L1, ..., L(N-1), GN]
            Mỗi Lk = Gk - expand(G(k+1));  phần tử cuối là residual (GN).
        """
        gp = self.gaussian_pyramid(img)
        lp = []

        for i in range(len(gp) - 1):
            h, w = gp[i].shape[:2]
            expanded = cv2.pyrUp(gp[i + 1], dstsize=(w, h))
            laplacian = gp[i] - expanded
            lp.append(laplacian)

        # Top level = residual (low-pass)
        lp.append(gp[-1])
        return lp

    def reconstruct_from_laplacian(self, lp: List[np.ndarray]) -> np.ndarray:
        """
        Tái tạo ảnh từ Laplacian pyramid.

        Parameters
        ----------
        lp : list[np.ndarray]
            Laplacian pyramid [L0, L1, ..., L(N-1), GN].

        Returns
        -------
        img : np.ndarray
            Ảnh tái tạo, dtype uint8, range [0, 255].
        """
        current = lp[-1].copy()  # Start from top (residual)

        for i in range(len(lp) - 2, -1, -1):
            h, w = lp[i].shape[:2]
            expanded = cv2.pyrUp(current, dstsize=(w, h))
            current = expanded + lp[i]

        return np.clip(current, 0, 255).astype(np.uint8)

    # ══════════════════════════════════════════════════════════════════════════
    #  Fusion
    # ══════════════════════════════════════════════════════════════════════════

    def fuse(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Fusion hai ảnh bằng Laplacian Pyramid.

        Parameters
        ----------
        img1, img2 : np.ndarray
            Hai ảnh grayscale cùng kích thước (H, W).

        Returns
        -------
        fused : np.ndarray
            Ảnh fusion, dtype uint8, range [0, 255].
        """
        img1, img2 = self._match_size(img1, img2)

        lp1 = self.laplacian_pyramid(img1)
        lp2 = self.laplacian_pyramid(img2)

        fused_lp = []
        n = len(lp1)

        for i in range(n):
            if i == n - 1:
                # Top level (residual / low-frequency)
                fused_lp.append(self._fuse_low(lp1[i], lp2[i]))
            else:
                # Bandpass levels
                fused_lp.append(self._fuse_bandpass(lp1[i], lp2[i]))

        return self.reconstruct_from_laplacian(fused_lp)

    # ── Fusion rules ─────────────────────────────────────────────────────────

    def _fuse_low(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fusion rule cho top level (residual / low-frequency)."""
        if self.low_fusion == 'average':
            return (a + b) / 2.0
        elif self.low_fusion == 'max':
            return np.maximum(a, b)
        return (a + b) / 2.0

    def _fuse_bandpass(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fusion rule cho bandpass (Laplacian) levels."""
        if self.fusion_mode == 'max_abs':
            mask = np.abs(a) >= np.abs(b)
            return np.where(mask, a, b)
        elif self.fusion_mode == 'average':
            return (a + b) / 2.0
        elif self.fusion_mode == 'weighted':
            return self._salience_weighted(a, b)
        return np.where(np.abs(a) >= np.abs(b), a, b)

    @staticmethod
    def _salience_weighted(
        a: np.ndarray, b: np.ndarray, ksize: int = 5
    ) -> np.ndarray:
        """
        Weighted fusion dựa trên local salience (absolute value smoothed).
        """
        kernel = np.ones((ksize, ksize), np.float64) / (ksize * ksize)

        sal_a = cv2.filter2D(np.abs(a), -1, kernel, borderType=cv2.BORDER_REFLECT)
        sal_b = cv2.filter2D(np.abs(b), -1, kernel, borderType=cv2.BORDER_REFLECT)

        total = sal_a + sal_b + 1e-10
        w_a = sal_a / total
        w_b = sal_b / total

        return w_a * a + w_b * b

    @staticmethod
    def _match_size(
        img1: np.ndarray, img2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resize img2 cho khớp kích thước img1."""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        return img1, img2


# ── Convenience functions ─────────────────────────────────────────────────────

def pyramid_fusion(
    img1: np.ndarray,
    img2: np.ndarray,
    levels: int = 5,
    fusion_mode: str = 'max_abs',
) -> np.ndarray:
    """
    Shortcut function cho Laplacian Pyramid fusion.

    >>> from algorithm.decomposition.pyramid_fusion import pyramid_fusion
    >>> fused = pyramid_fusion(img1, img2, levels=5)
    """
    fuser = PyramidFusion(levels=levels, fusion_mode=fusion_mode)
    return fuser.fuse(img1, img2)
