"""
Non-subsampled Shearlet Transform (NSST) Image Fusion
=====================================================

Thuật toán NSST cho image fusion — shift-invariant, multi-scale,
multi-directional decomposition:

- Sử dụng Non-subsampled Laplacian Pyramid (NSLP) cho multi-scale decomposition
- Sử dụng Shearing filters cho directional decomposition
- Không downsampling → shift-invariant → tránh artifact Gibbs

Pipeline:
    Input Image
    → NSLP decompose (multi-scale) → low-freq + [highpass_1, ..., highpass_J]
    → Shearing filter (multi-direction) trên mỗi highpass_j
    → Fusion rule
    → Inverse Shearing → Inverse NSLP
    → Fused Image

References:
    - Easley, G., Labate, D., & Lim, W.-Q. (2008). Sparse directional image
      representations using the discrete shearlet transform. Applied and
      Computational Harmonic Analysis.
    - Liu, Y., Liu, S., & Wang, Z. (2015). A general framework for image fusion
      based on multi-scale transform and sparse representation. Information Fusion.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

from ._base import BaseFusion


class NSSTFusion(BaseFusion):
    """
    Non-subsampled Shearlet Transform Fusion.

    Parameters
    ----------
    levels : int
        Số level phân tách multi-scale (default=4).
    directions : list[int]
        Số hướng (directions) tại mỗi level. Mặc định [8, 8, 16, 16].
        Phải có len(directions) == levels.
    fusion_mode : str
        Chế độ fusion cho high-frequency coefficients:
        - 'max_abs'   : chọn hệ số có |value| lớn hơn (mặc định)
        - 'regional'  : regional energy-based selection
    low_fusion : str
        Chế độ fusion cho low-frequency:
        - 'average'   : trung bình (mặc định)
        - 'max'       : chọn pixel lớn hơn

    Example
    -------
    >>> from algorithm.decomposition import NSSTFusion
    >>> fuser = NSSTFusion(levels=4, directions=[8, 8, 16, 16])
    >>> fused_gray = fuser.fuse(img1_gray, img2_gray)       # grayscale
    >>> fused_color = fuser.fuse_ycbcr(img1_bgr, img2_bgr)  # YCbCr color
    >>> fused_auto = fuser.fuse_auto(img1, img2)             # auto-detect
    """

    def __init__(
        self,
        levels: int = 4,
        directions: Optional[List[int]] = None,
        fusion_mode: str = 'max_abs',
        low_fusion: str = 'average',
    ):
        self.levels = levels
        self.directions = directions or [8] * levels
        if len(self.directions) != levels:
            raise ValueError(
                f"len(directions)={len(self.directions)} != levels={levels}"
            )
        self.fusion_mode = fusion_mode
        self.low_fusion = low_fusion

    # ══════════════════════════════════════════════════════════════════════════
    #  Non-subsampled Laplacian Pyramid (NSLP) — Multi-scale decomposition
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _nslp_filter(level: int) -> np.ndarray:
        """
        Tạo lowpass filter cho NSLP tại level cho trước.
        Sử dụng 'à trous' algorithm — upsample filter bằng cách chèn zeros.
        """
        # Base filter: [1, 4, 6, 4, 1] / 16 (binomial filter)
        h0 = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0

        # Upsample bằng cách chèn (2^level - 1) zeros giữa các phần tử
        step = 2 ** level
        h_up = np.zeros(1 + (len(h0) - 1) * step, dtype=np.float64)
        h_up[::step] = h0

        # 2D filter = outer product
        return np.outer(h_up, h_up)

    def nslp_decompose(self, img: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Non-subsampled Laplacian Pyramid decomposition.

        Returns
        -------
        low : np.ndarray
            Low-frequency approximation (kích thước giữ nguyên).
        highs : list[np.ndarray]
            High-frequency details tại mỗi level.
        """
        current = img.astype(np.float64)
        highs = []

        for j in range(self.levels):
            h2d = self._nslp_filter(j)
            low = cv2.filter2D(current, -1, h2d, borderType=cv2.BORDER_REFLECT)
            high = current - low
            highs.append(high)
            current = low

        return current, highs

    def nslp_reconstruct(
        self, low: np.ndarray, highs: List[np.ndarray]
    ) -> np.ndarray:
        """
        Tái tạo ảnh từ NSLP: đơn giản cộng lại tất cả bandpass + lowpass.
        """
        rec = low.copy()
        for high in reversed(highs):
            rec = rec + high
        return rec

    # ══════════════════════════════════════════════════════════════════════════
    #  Shearing Filters — Multi-directional decomposition
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _make_shear_filters(n_directions: int, size: int = 17) -> List[np.ndarray]:
        """
        Tạo bộ directional bandpass filters thông qua oriented Gabor-like filters.

        Parameters
        ----------
        n_directions : int
            Số hướng cần tạo (ví dụ 8, 16).
        size : int
            Kích thước filter (nên lẻ).

        Returns
        -------
        filters : list[np.ndarray]
            List các 2D filters, mỗi filter tương ứng 1 hướng.
        """
        filters = []
        half = size // 2
        y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
        r = np.sqrt(x ** 2 + y ** 2) + 1e-10

        # Bandpass radial envelope
        sigma = size / 4.0
        radial = np.exp(-r ** 2 / (2 * sigma ** 2))
        radial[half, half] = 0  # remove DC

        for k in range(n_directions):
            angle = np.pi * k / n_directions
            cos_a, sin_a = np.cos(angle), np.sin(angle)

            # Project onto direction
            proj = x * cos_a + y * sin_a
            perp = -x * sin_a + y * cos_a

            # Directional Gaussian envelope
            sigma_par = sigma * 1.5
            sigma_perp = sigma * 0.4
            directional = np.exp(
                -proj ** 2 / (2 * sigma_par ** 2)
                - perp ** 2 / (2 * sigma_perp ** 2)
            )

            f = radial * directional
            f -= f.mean()  # zero-mean
            norm = np.sqrt(np.sum(f ** 2))
            if norm > 1e-10:
                f /= norm
            filters.append(f)

        return filters

    def shear_decompose(
        self, high: np.ndarray, n_directions: int
    ) -> List[np.ndarray]:
        """
        Directional decomposition của 1 highpass sub-band.

        Returns
        -------
        dir_coeffs : list[np.ndarray]
            Hệ số theo mỗi hướng, kích thước giữ nguyên.
        """
        filters = self._make_shear_filters(n_directions, size=17)
        dir_coeffs = []
        for f in filters:
            coeff = cv2.filter2D(
                high, -1, f, borderType=cv2.BORDER_REFLECT
            )
            dir_coeffs.append(coeff)
        return dir_coeffs

    def shear_reconstruct(
        self, dir_coeffs: List[np.ndarray], n_directions: int
    ) -> np.ndarray:
        """
        Tái tạo highpass sub-band từ directional coefficients.
        Cộng tổng tất cả hướng và normalize.
        """
        rec = np.sum(dir_coeffs, axis=0)
        # Normalize để năng lượng phù hợp
        rec /= max(n_directions / 4.0, 1.0)
        return rec

    # ══════════════════════════════════════════════════════════════════════════
    #  Full NSST Decompose / Reconstruct
    # ══════════════════════════════════════════════════════════════════════════

    def decompose(self, img: np.ndarray) -> dict:
        """
        Full NSST decomposition.

        Returns
        -------
        result : dict
            {
                'low': np.ndarray,              # Low-frequency approximation
                'shear_coeffs': [                # list of levels
                    [dir_0, dir_1, ..., dir_k],  # directional coeffs per level
                    ...
                ],
                'highs': [high_0, ..., high_J],  # NSLP highpass bands (for reconstruction)
            }
        """
        low, highs = self.nslp_decompose(img)

        all_shear_coeffs = []
        for j, high_j in enumerate(highs):
            n_dir = self.directions[j]
            dir_coeffs = self.shear_decompose(high_j, n_dir)
            all_shear_coeffs.append(dir_coeffs)

        return {
            'low': low,
            'shear_coeffs': all_shear_coeffs,
            'highs': highs,
        }

    def reconstruct(self, coeffs: dict) -> np.ndarray:
        """
        Tái tạo ảnh từ NSST coefficients.
        """
        # Tái tạo mỗi highpass band từ directional coefficients
        rec_highs = []
        for j, dir_coeffs in enumerate(coeffs['shear_coeffs']):
            n_dir = self.directions[j]
            rec_high = self.shear_reconstruct(dir_coeffs, n_dir)
            rec_highs.append(rec_high)

        rec = self.nslp_reconstruct(coeffs['low'], rec_highs)
        return np.clip(rec, 0, 255).astype(np.uint8)

    # ══════════════════════════════════════════════════════════════════════════
    #  Fusion
    # ══════════════════════════════════════════════════════════════════════════

    def fuse(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Fusion hai ảnh bằng NSST.

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

        c1 = self.decompose(img1)
        c2 = self.decompose(img2)

        # Fuse low-frequency
        fused_low = self._fuse_low(c1['low'], c2['low'])

        # Fuse directional coefficients tại mỗi level
        fused_shear = []
        for j in range(self.levels):
            fused_dirs = []
            for d1, d2 in zip(c1['shear_coeffs'][j], c2['shear_coeffs'][j]):
                fused_dirs.append(self._fuse_high(d1, d2))
            fused_shear.append(fused_dirs)

        fused_coeffs = {
            'low': fused_low,
            'shear_coeffs': fused_shear,
        }

        return self.reconstruct(fused_coeffs)

    # ── Fusion rules ─────────────────────────────────────────────────────────

    def _fuse_low(self, ll1: np.ndarray, ll2: np.ndarray) -> np.ndarray:
        """Fusion rule cho low-frequency."""
        if self.low_fusion == 'average':
            return (ll1 + ll2) / 2.0
        elif self.low_fusion == 'max':
            return np.maximum(ll1, ll2)
        return (ll1 + ll2) / 2.0

    def _fuse_high(self, d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
        """Fusion rule cho directional high-frequency coefficients."""
        if self.fusion_mode == 'max_abs':
            mask = np.abs(d1) >= np.abs(d2)
            return np.where(mask, d1, d2)
        elif self.fusion_mode == 'regional':
            return self._regional_energy_fusion(d1, d2)
        return np.where(np.abs(d1) >= np.abs(d2), d1, d2)

    @staticmethod
    def _regional_energy_fusion(
        d1: np.ndarray, d2: np.ndarray, ksize: int = 7
    ) -> np.ndarray:
        """
        Regional energy-based fusion: chọn hệ số từ ảnh có local energy lớn hơn.
        Sử dụng cửa sổ trượt ksize × ksize để tính energy.
        """
        kernel = np.ones((ksize, ksize), dtype=np.float64) / (ksize * ksize)
        e1 = cv2.filter2D(d1 ** 2, -1, kernel, borderType=cv2.BORDER_REFLECT)
        e2 = cv2.filter2D(d2 ** 2, -1, kernel, borderType=cv2.BORDER_REFLECT)
        mask = e1 >= e2
        return np.where(mask, d1, d2)

    @staticmethod
    def _match_size(
        img1: np.ndarray, img2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resize img2 cho khớp kích thước img1."""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        return img1, img2


# ── Convenience function ──────────────────────────────────────────────────────

def nsst_fusion(
    img1: np.ndarray,
    img2: np.ndarray,
    levels: int = 4,
    directions: Optional[List[int]] = None,
    fusion_mode: str = 'max_abs',
) -> np.ndarray:
    """
    Shortcut function cho NSST fusion.

    >>> from algorithm.decomposition.nsst_fusion import nsst_fusion
    >>> fused = nsst_fusion(img1, img2, levels=4, directions=[8, 8, 16, 16])
    """
    fuser = NSSTFusion(
        levels=levels, directions=directions, fusion_mode=fusion_mode
    )
    return fuser.fuse(img1, img2)
