"""
Guided Filter & Edge-Preserving Decomposition Image Fusion
==========================================================

Thuật toán sử dụng Guided Filter để phân tách ảnh thành:
- Base layer (low-frequency, structural content)
- Detail layer (high-frequency, edges & textures)

Ưu điểm so với bilateral filter:
- Không bị gradient reversal artifact
- O(N) complexity (khác bilateral O(N·r²))
- Giữ edge tốt hơn → phù hợp medical image fusion

Pipeline:
    Input Image
    → Guided Filter → Base layer (edge-preserving smooth)
    → Detail = Original - Base
    → Fusion rule riêng cho Base (average/weighted) và Detail (max_abs/weighted)
    → Fused = Fused_Base + Fused_Detail

Hỗ trợ thêm:
- Multi-scale guided decomposition (cascaded guided filter)
- Weight map-based fusion (cho cả base và detail)
- Rolling Guidance Filter variant

References:
    - He, K., Sun, J., & Tang, X. (2013). Guided image filtering. IEEE TPAMI.
    - Li, S., Kang, X., & Hu, J. (2013). Image fusion with guided filtering.
      IEEE Trans. Image Processing.
    - Zhang, Q., Shen, X., Xu, L., & Jia, J. (2014). Rolling guidance filter. ECCV.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional

from ._base import BaseFusion


class GuidedFilterFusion(BaseFusion):
    """
    Guided Filter & Edge-Preserving Decomposition Fusion.

    Parameters
    ----------
    radius : int
        Bán kính cửa sổ guided filter (default=8).
    eps : float
        Regularization parameter (default=0.01).
        eps nhỏ → giữ edge mạnh; eps lớn → smooth nhiều hơn.
    n_scales : int
        Số scale cho multi-scale decomposition (default=1).
        n_scales=1 → 2 layers (base + detail).
        n_scales>1 → cascaded guided filter → nhiều detail layers.
    detail_fusion : str
        Fusion rule cho detail layer(s):
        - 'max_abs'    : chọn pixel có |value| lớn hơn (mặc định)
        - 'weighted'   : weighted average based on local saliency
        - 'average'    : simple average
    base_fusion : str
        Fusion rule cho base layer:
        - 'weighted'   : weighted average dựa trên local contrast (mặc định)
        - 'average'    : simple average
        - 'max'        : chọn pixel lớn hơn

    Example
    -------
    >>> from algorithm.decomposition import GuidedFilterFusion
    >>> fuser = GuidedFilterFusion(radius=8, eps=0.01, n_scales=3)
    >>> fused_gray = fuser.fuse(img1_gray, img2_gray)       # grayscale
    >>> fused_color = fuser.fuse_ycbcr(img1_bgr, img2_bgr)  # YCbCr color
    >>> fused_auto = fuser.fuse_auto(img1, img2)             # auto-detect
    """

    def __init__(
        self,
        radius: int = 8,
        eps: float = 0.01,
        n_scales: int = 1,
        detail_fusion: str = 'max_abs',
        base_fusion: str = 'weighted',
    ):
        self.radius = radius
        self.eps = eps
        self.n_scales = n_scales
        self.detail_fusion = detail_fusion
        self.base_fusion = base_fusion

    # ══════════════════════════════════════════════════════════════════════════
    #  Guided Filter — Core implementation
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def guided_filter(
        I: np.ndarray,
        p: np.ndarray,
        radius: int,
        eps: float,
    ) -> np.ndarray:
        """
        Guided Image Filter (He et al., 2013).

        Parameters
        ----------
        I : np.ndarray
            Guidance image (float64, range [0, 1]).
        p : np.ndarray
            Input image to be filtered (float64, range [0, 1]).
        radius : int
            Bán kính cửa sổ (window size = 2*radius + 1).
        eps : float
            Regularization parameter.

        Returns
        -------
        q : np.ndarray
            Filtered output.

        Notes
        -----
        q_i = a_k * I_i + b_k,  ∀ i ∈ ω_k
        a_k = (1/|ω|) Σ(I_i * p_i) - μ_k * p̄_k) / (σ²_k + eps)
        b_k = p̄_k - a_k * μ_k
        """
        ksize = 2 * radius + 1

        mean_I = cv2.boxFilter(I, -1, (ksize, ksize), borderType=cv2.BORDER_REFLECT)
        mean_p = cv2.boxFilter(p, -1, (ksize, ksize), borderType=cv2.BORDER_REFLECT)
        mean_Ip = cv2.boxFilter(I * p, -1, (ksize, ksize), borderType=cv2.BORDER_REFLECT)
        mean_II = cv2.boxFilter(I * I, -1, (ksize, ksize), borderType=cv2.BORDER_REFLECT)

        cov_Ip = mean_Ip - mean_I * mean_p  # covariance
        var_I = mean_II - mean_I * mean_I     # variance

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, -1, (ksize, ksize), borderType=cv2.BORDER_REFLECT)
        mean_b = cv2.boxFilter(b, -1, (ksize, ksize), borderType=cv2.BORDER_REFLECT)

        q = mean_a * I + mean_b
        return q

    # ══════════════════════════════════════════════════════════════════════════
    #  Edge-Preserving Decomposition
    # ══════════════════════════════════════════════════════════════════════════

    def decompose(self, img: np.ndarray) -> dict:
        """
        Phân tách ảnh thành base + detail layers bằng guided filter.

        Parameters
        ----------
        img : np.ndarray
            Ảnh grayscale (H, W), dtype uint8 hoặc float.

        Returns
        -------
        result : dict
            {
                'base': np.ndarray,            # Base layer (cuối cùng)
                'details': [d1, d2, ..., dN],  # Detail layers (từ fine → coarse)
            }
        """
        img_float = img.astype(np.float64) / 255.0

        details = []
        current = img_float.copy()

        for s in range(self.n_scales):
            # Tăng radius và eps theo scale
            r = self.radius * (2 ** s)
            e = self.eps * (4 ** s)

            # Guided filter: dùng chính ảnh hiện tại làm guidance
            base = self.guided_filter(current, current, r, e)
            detail = current - base
            details.append(detail)
            current = base

        return {
            'base': current,
            'details': details,
        }

    def reconstruct(self, coeffs: dict) -> np.ndarray:
        """
        Tái tạo ảnh từ base + detail layers.
        """
        rec = coeffs['base'].copy()
        for detail in reversed(coeffs['details']):
            rec = rec + detail

        rec = np.clip(rec * 255.0, 0, 255).astype(np.uint8)
        return rec

    # ══════════════════════════════════════════════════════════════════════════
    #  Weight Maps
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _saliency_map(img_float: np.ndarray, ksize: int = 5) -> np.ndarray:
        """
        Tính saliency map dựa trên Laplacian (edge detection).
        """
        lap = cv2.Laplacian(img_float, cv2.CV_64F)
        sal = np.abs(lap)
        # Smooth saliency
        sal = cv2.GaussianBlur(sal, (ksize, ksize), 0)
        return sal

    @staticmethod
    def _local_contrast(img_float: np.ndarray, ksize: int = 11) -> np.ndarray:
        """
        Tính local contrast map (standard deviation cục bộ).
        """
        mean = cv2.boxFilter(img_float, -1, (ksize, ksize), borderType=cv2.BORDER_REFLECT)
        mean_sq = cv2.boxFilter(img_float ** 2, -1, (ksize, ksize), borderType=cv2.BORDER_REFLECT)
        var = mean_sq - mean ** 2
        var = np.maximum(var, 0)  # tránh âm do numerical error
        return np.sqrt(var)

    def _compute_weight_maps(
        self, img1_float: np.ndarray, img2_float: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tính weight maps cho 2 ảnh dựa trên saliency.
        Weight maps được refine bằng guided filter để giữ edge-consistent.

        Returns
        -------
        w1, w2 : np.ndarray
            Weight maps, w1 + w2 = 1 tại mỗi pixel.
        """
        s1 = self._saliency_map(img1_float)
        s2 = self._saliency_map(img2_float)

        # Binary weight: pixel nào có saliency cao hơn nhận trọng số 1
        p1 = (s1 >= s2).astype(np.float64)
        p2 = 1.0 - p1

        # Refine weight maps bằng guided filter → smooth + giữ edges
        w1 = self.guided_filter(img1_float, p1, self.radius, self.eps)
        w2 = self.guided_filter(img2_float, p2, self.radius, self.eps)

        # Normalize
        total = w1 + w2 + 1e-10
        w1 = w1 / total
        w2 = w2 / total

        return w1, w2

    # ══════════════════════════════════════════════════════════════════════════
    #  Fusion
    # ══════════════════════════════════════════════════════════════════════════

    def fuse(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Fusion hai ảnh bằng Guided Filter Decomposition.

        Parameters
        ----------
        img1, img2 : np.ndarray
            Hai ảnh grayscale, cùng kích thước (H, W).

        Returns
        -------
        fused : np.ndarray
            Ảnh fusion, dtype uint8, range [0, 255].
        """
        img1, img2 = self._match_size(img1, img2)

        c1 = self.decompose(img1)
        c2 = self.decompose(img2)

        # Fuse base layer
        fused_base = self._fuse_base(c1['base'], c2['base'], img1, img2)

        # Fuse detail layers
        fused_details = []
        for d1, d2 in zip(c1['details'], c2['details']):
            fused_details.append(self._fuse_detail(d1, d2))

        fused_coeffs = {
            'base': fused_base,
            'details': fused_details,
        }

        return self.reconstruct(fused_coeffs)

    def fuse_with_weight_maps(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> np.ndarray:
        """
        Fusion bằng guided filter weight maps (Li et al., 2013).

        Thay vì phân tách base/detail riêng, dùng weight maps được refine
        bằng guided filter để blend 2 ảnh trực tiếp trên mỗi level.
        """
        img1, img2 = self._match_size(img1, img2)

        c1 = self.decompose(img1)
        c2 = self.decompose(img2)

        img1_f = img1.astype(np.float64) / 255.0
        img2_f = img2.astype(np.float64) / 255.0

        w1, w2 = self._compute_weight_maps(img1_f, img2_f)

        # Weighted blend cho base
        fused_base = w1 * c1['base'] + w2 * c2['base']

        # Weighted blend cho detail
        fused_details = []
        for d1, d2 in zip(c1['details'], c2['details']):
            fused_details.append(w1 * d1 + w2 * d2)

        fused_coeffs = {
            'base': fused_base,
            'details': fused_details,
        }

        return self.reconstruct(fused_coeffs)

    # ── Fusion rules ─────────────────────────────────────────────────────────

    def _fuse_base(
        self,
        b1: np.ndarray,
        b2: np.ndarray,
        img1: np.ndarray,
        img2: np.ndarray,
    ) -> np.ndarray:
        """Fusion rule cho base layer."""
        if self.base_fusion == 'average':
            return (b1 + b2) / 2.0
        elif self.base_fusion == 'max':
            return np.maximum(b1, b2)
        elif self.base_fusion == 'weighted':
            # Weighted average dựa trên local contrast
            img1_f = img1.astype(np.float64) / 255.0
            img2_f = img2.astype(np.float64) / 255.0
            c1 = self._local_contrast(img1_f)
            c2 = self._local_contrast(img2_f)
            total = c1 + c2 + 1e-10
            w1 = c1 / total
            w2 = c2 / total
            return w1 * b1 + w2 * b2
        return (b1 + b2) / 2.0

    def _fuse_detail(self, d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
        """Fusion rule cho detail layer(s)."""
        if self.detail_fusion == 'max_abs':
            mask = np.abs(d1) >= np.abs(d2)
            return np.where(mask, d1, d2)
        elif self.detail_fusion == 'average':
            return (d1 + d2) / 2.0
        elif self.detail_fusion == 'weighted':
            sal1 = np.abs(d1)
            sal2 = np.abs(d2)
            total = sal1 + sal2 + 1e-10
            w1 = sal1 / total
            w2 = sal2 / total
            return w1 * d1 + w2 * d2
        return np.where(np.abs(d1) >= np.abs(d2), d1, d2)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _match_size(
        img1: np.ndarray, img2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resize img2 cho khớp kích thước img1."""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        return img1, img2


# ══════════════════════════════════════════════════════════════════════════════
#  Rolling Guidance Filter (Zhang et al., 2014)
# ══════════════════════════════════════════════════════════════════════════════

class RollingGuidanceFilter:
    """
    Rolling Guidance Filter — edge-preserving smoothing cải tiến.
    Lặp lại guided filter nhiều lần, mỗi lần dùng kết quả trước làm guidance.
    Hiệu quả loại bỏ texture nhỏ nhưng giữ edge lớn.

    Parameters
    ----------
    radius : int
        Bán kính cửa sổ (default=4).
    eps : float
        Regularization parameter (default=0.01).
    iterations : int
        Số lần lặp (default=4). Nhiều hơn → smooth hơn nhưng giữ edges.

    Example
    -------
    >>> rgf = RollingGuidanceFilter(radius=4, eps=0.01, iterations=4)
    >>> smoothed = rgf.filter(img)
    """

    def __init__(self, radius: int = 4, eps: float = 0.01, iterations: int = 4):
        self.radius = radius
        self.eps = eps
        self.iterations = iterations

    def filter(self, img: np.ndarray) -> np.ndarray:
        """
        Áp dụng Rolling Guidance Filter.

        Parameters
        ----------
        img : np.ndarray
            Ảnh grayscale (H, W), dtype uint8 hoặc float.

        Returns
        -------
        result : np.ndarray
            Ảnh đã smoothing, dtype uint8.
        """
        img_float = img.astype(np.float64) / 255.0

        # Iteration 0: Gaussian blur as initial guidance
        ksize = 2 * self.radius + 1
        guidance = cv2.GaussianBlur(img_float, (ksize, ksize), 0)

        for _ in range(self.iterations):
            guidance = GuidedFilterFusion.guided_filter(
                guidance, img_float, self.radius, self.eps
            )

        return np.clip(guidance * 255.0, 0, 255).astype(np.uint8)


# ── Convenience function ──────────────────────────────────────────────────────

def guided_filter_fusion(
    img1: np.ndarray,
    img2: np.ndarray,
    radius: int = 8,
    eps: float = 0.01,
    n_scales: int = 1,
    detail_fusion: str = 'max_abs',
    base_fusion: str = 'weighted',
) -> np.ndarray:
    """
    Shortcut function cho Guided Filter fusion.

    >>> from algorithm.decomposition.guided_filter_fusion import guided_filter_fusion
    >>> fused = guided_filter_fusion(img1, img2, radius=8, eps=0.01)
    """
    fuser = GuidedFilterFusion(
        radius=radius, eps=eps, n_scales=n_scales,
        detail_fusion=detail_fusion, base_fusion=base_fusion,
    )
    return fuser.fuse(img1, img2)
