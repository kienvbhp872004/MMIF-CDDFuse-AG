"""
Discrete Wavelet Transform (DWT) Image Fusion
==============================================

Thuật toán DWT multi-level cho image fusion:
- Phân tách ảnh thành các sub-band tần số (LL, LH, HL, HH) qua nhiều level
- Fusion rule: trung bình cho low-frequency (LL), max absolute cho high-frequency
- Hỗ trợ nhiều loại wavelet: haar, db2, db4, sym4, bior1.3, coif1, ...
- Hỗ trợ fusion rule tuỳ chỉnh

References:
    - Pajares, G., & de la Cruz, J. M. (2004). A wavelet-based image fusion tutorial.
    - Li, H., Manjunath, B. S., & Mitra, S. K. (1995). Multisensor image fusion using
      the wavelet transform. Graphical Models and Image Processing.
"""

import cv2
import numpy as np
import pywt
from typing import Tuple, Optional, Callable

from ._base import BaseFusion


class DWTFusion(BaseFusion):
    """
    Multi-level Discrete Wavelet Transform Fusion.

    Parameters
    ----------
    wavelet : str
        Tên wavelet (default='db2'). Hỗ trợ: 'haar', 'db2', 'db4', 'sym4',
        'bior1.3', 'coif1', v.v.
    level : int
        Số level phân tách (default=3). Level càng cao → tách chi tiết càng sâu.
    fusion_mode : str
        Chế độ fusion cho high-frequency sub-bands:
        - 'max_abs'    : chọn hệ số có giá trị tuyệt đối lớn hơn (mặc định)
        - 'average'    : trung bình hai hệ số
        - 'weighted'   : trung bình có trọng số dựa trên energy cục bộ
    low_fusion : str
        Chế độ fusion cho low-frequency (LL) sub-band:
        - 'average'    : trung bình (mặc định)
        - 'max'        : chọn pixel lớn hơn
        - 'weighted'   : trung bình có trọng số theo variance cục bộ

    Example
    -------
    >>> from algorithm.decomposition import DWTFusion
    >>> fuser = DWTFusion(wavelet='db4', level=3, fusion_mode='max_abs')
    >>> fused_gray = fuser.fuse(img1_gray, img2_gray)       # grayscale
    >>> fused_color = fuser.fuse_ycbcr(img1_bgr, img2_bgr)  # YCbCr color
    >>> fused_auto = fuser.fuse_auto(img1, img2)             # auto-detect
    """

    SUPPORTED_FUSION = ('max_abs', 'average', 'weighted')
    SUPPORTED_LOW    = ('average', 'max', 'weighted')

    def __init__(
        self,
        wavelet: str = 'db2',
        level: int = 3,
        fusion_mode: str = 'max_abs',
        low_fusion: str = 'average',
    ):
        if fusion_mode not in self.SUPPORTED_FUSION:
            raise ValueError(f"fusion_mode must be one of {self.SUPPORTED_FUSION}")
        if low_fusion not in self.SUPPORTED_LOW:
            raise ValueError(f"low_fusion must be one of {self.SUPPORTED_LOW}")

        self.wavelet = wavelet
        self.level = level
        self.fusion_mode = fusion_mode
        self.low_fusion = low_fusion

    # ── Core API ──────────────────────────────────────────────────────────────

    def decompose(self, img: np.ndarray) -> list:
        """
        Phân tách ảnh bằng DWT multi-level.

        Parameters
        ----------
        img : np.ndarray
            Ảnh grayscale (H, W), dtype float64.

        Returns
        -------
        coeffs : list
            Danh sách hệ số [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]
            theo format pywt.wavedec2.
        """
        img_float = img.astype(np.float64)
        coeffs = pywt.wavedec2(img_float, self.wavelet, level=self.level)
        return coeffs

    def reconstruct(self, coeffs: list) -> np.ndarray:
        """
        Tái tạo ảnh từ hệ số DWT.

        Parameters
        ----------
        coeffs : list
            Hệ số DWT theo format pywt.

        Returns
        -------
        img : np.ndarray
            Ảnh tái tạo, chuẩn hoá về [0, 255], dtype uint8.
        """
        rec = pywt.waverec2(coeffs, self.wavelet)
        rec = np.clip(rec, 0, 255).astype(np.uint8)
        return rec

    def fuse(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Fusion hai ảnh bằng DWT multi-level.

        Parameters
        ----------
        img1, img2 : np.ndarray
            Hai ảnh grayscale cùng kích thước (H, W).

        Returns
        -------
        fused : np.ndarray
            Ảnh kết quả fusion, dtype uint8, range [0, 255].
        """
        img1, img2 = self._match_size(img1, img2)

        coeffs1 = self.decompose(img1)
        coeffs2 = self.decompose(img2)

        # Fuse low-frequency (approximation) sub-band
        fused_coeffs = [self._fuse_low(coeffs1[0], coeffs2[0])]

        # Fuse high-frequency detail sub-bands tại mỗi level
        for detail1, detail2 in zip(coeffs1[1:], coeffs2[1:]):
            fused_detail = tuple(
                self._fuse_high(d1, d2) for d1, d2 in zip(detail1, detail2)
            )
            fused_coeffs.append(fused_detail)

        return self.reconstruct(fused_coeffs)

    # ── Fusion rules ─────────────────────────────────────────────────────────

    def _fuse_low(self, ll1: np.ndarray, ll2: np.ndarray) -> np.ndarray:
        """Fusion rule cho low-frequency sub-band."""
        if self.low_fusion == 'average':
            return (ll1 + ll2) / 2.0
        elif self.low_fusion == 'max':
            return np.maximum(ll1, ll2)
        elif self.low_fusion == 'weighted':
            return self._weighted_average(ll1, ll2, ksize=7)
        return (ll1 + ll2) / 2.0

    def _fuse_high(self, d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
        """Fusion rule cho high-frequency sub-bands (LH, HL, HH)."""
        if self.fusion_mode == 'max_abs':
            mask = np.abs(d1) >= np.abs(d2)
            return np.where(mask, d1, d2)
        elif self.fusion_mode == 'average':
            return (d1 + d2) / 2.0
        elif self.fusion_mode == 'weighted':
            return self._weighted_average(d1, d2, ksize=5)
        return np.where(np.abs(d1) >= np.abs(d2), d1, d2)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _weighted_average(
        a: np.ndarray, b: np.ndarray, ksize: int = 5
    ) -> np.ndarray:
        """
        Trung bình có trọng số dựa trên local energy (variance cục bộ).
        Vùng có energy cao hơn → trọng số lớn hơn.
        """
        kernel = np.ones((ksize, ksize), dtype=np.float64) / (ksize * ksize)

        e_a = cv2.filter2D(a.astype(np.float64) ** 2, -1, kernel)
        e_b = cv2.filter2D(b.astype(np.float64) ** 2, -1, kernel)

        total = e_a + e_b + 1e-10  # tránh chia 0
        w_a = e_a / total
        w_b = e_b / total

        return w_a * a + w_b * b

    @staticmethod
    def _match_size(
        img1: np.ndarray, img2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resize img2 cho khớp kích thước img1 nếu khác nhau."""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        return img1, img2


# ── Convenience function ──────────────────────────────────────────────────────

def dwt_fusion(
    img1: np.ndarray,
    img2: np.ndarray,
    wavelet: str = 'db2',
    level: int = 3,
    fusion_mode: str = 'max_abs',
) -> np.ndarray:
    """
    Shortcut function cho DWT fusion.

    >>> from algorithm.decomposition.dwt_fusion import dwt_fusion
    >>> fused = dwt_fusion(img1, img2, wavelet='db4', level=3)
    """
    fuser = DWTFusion(wavelet=wavelet, level=level, fusion_mode=fusion_mode)
    return fuser.fuse(img1, img2)
