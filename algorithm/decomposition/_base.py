"""
Base class cho tất cả Decomposition Fusion algorithms.
Cung cấp hỗ trợ fusion trên không gian màu YCbCr.

Pipeline YCbCr:
    Input (BGR) → YCbCr
    → Fuse kênh Y (luminance) bằng thuật toán decomposition
    → Fuse kênh Cb, Cr (chrominance) bằng weighted average
    → YCbCr → BGR output
"""

import cv2
import numpy as np
from typing import Tuple


class BaseFusion:
    """
    Base class cung cấp chức năng fusion trên không gian màu YCbCr.

    Subclass chỉ cần implement `fuse(img1_gray, img2_gray) -> fused_gray`
    cho ảnh grayscale. Base class sẽ tự động xử lý:
    - Phát hiện ảnh màu (3 channels)
    - Chuyển đổi BGR ↔ YCbCr
    - Fusion Y channel bằng thuật toán của subclass
    - Fusion Cb/Cr channels bằng rule riêng
    """

    # Chrominance fusion mode: 'average', 'weighted', 'source1', 'source2'
    chroma_fusion: str = 'weighted'

    def fuse_ycbcr(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Fusion hai ảnh màu trong không gian YCbCr.

        Parameters
        ----------
        img1, img2 : np.ndarray
            Hai ảnh BGR (H, W, 3), dtype uint8.

        Returns
        -------
        fused : np.ndarray
            Ảnh fusion màu BGR, dtype uint8.
        """
        img1, img2 = self._match_size_color(img1, img2)

        # Chuyển BGR → YCbCr
        ycbcr1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)
        ycbcr2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)

        y1, cr1, cb1 = cv2.split(ycbcr1)
        y2, cr2, cb2 = cv2.split(ycbcr2)

        # Fuse kênh Y (luminance) bằng thuật toán decomposition của subclass
        y_fused = self.fuse(y1, y2)

        # Fuse kênh Cb, Cr (chrominance)
        cr_fused = self._fuse_chrominance(cr1, cr2, y1, y2)
        cb_fused = self._fuse_chrominance(cb1, cb2, y1, y2)

        # Ghép lại và chuyển YCbCr → BGR
        fused_ycbcr = cv2.merge([y_fused, cr_fused, cb_fused])
        fused_bgr = cv2.cvtColor(fused_ycbcr, cv2.COLOR_YCrCb2BGR)

        return fused_bgr

    def fuse_auto(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Tự động phát hiện ảnh grayscale hay màu và chọn pipeline phù hợp.

        - Grayscale (H, W) → fuse() trực tiếp
        - Color (H, W, 3) → fuse_ycbcr() trong không gian YCbCr
        """
        if self._is_color(img1) or self._is_color(img2):
            # Đảm bảo cả hai đều là color
            img1 = self._ensure_color(img1)
            img2 = self._ensure_color(img2)
            return self.fuse_ycbcr(img1, img2)
        else:
            return self.fuse(img1, img2)

    # ── Chrominance fusion rules ─────────────────────────────────────────────

    def _fuse_chrominance(
        self,
        c1: np.ndarray,
        c2: np.ndarray,
        y1: np.ndarray,
        y2: np.ndarray,
    ) -> np.ndarray:
        """
        Fusion rule cho kênh chrominance (Cb hoặc Cr).

        Parameters
        ----------
        c1, c2 : np.ndarray
            Kênh chrominance từ 2 ảnh nguồn, dtype uint8.
        y1, y2 : np.ndarray
            Kênh luminance (Y) từ 2 ảnh nguồn — dùng cho weighted mode.

        Returns
        -------
        fused : np.ndarray
            Kênh chrominance đã fusion, dtype uint8.
        """
        if self.chroma_fusion == 'average':
            return self._chroma_average(c1, c2)
        elif self.chroma_fusion == 'weighted':
            return self._chroma_weighted(c1, c2, y1, y2)
        elif self.chroma_fusion == 'source1':
            return c1.copy()
        elif self.chroma_fusion == 'source2':
            return c2.copy()
        return self._chroma_weighted(c1, c2, y1, y2)

    @staticmethod
    def _chroma_average(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
        """Trung bình đơn giản cho chrominance."""
        return ((c1.astype(np.float64) + c2.astype(np.float64)) / 2.0
                ).clip(0, 255).astype(np.uint8)

    @staticmethod
    def _chroma_weighted(
        c1: np.ndarray,
        c2: np.ndarray,
        y1: np.ndarray,
        y2: np.ndarray,
        ksize: int = 7,
    ) -> np.ndarray:
        """
        Weighted average cho chrominance dựa trên luminance saliency.

        Vùng có luminance contrast cao hơn → chrominance từ ảnh đó
        được ưu tiên hơn (giữ màu nhất quán với cấu trúc).
        """
        # Tính local variance của Y channel làm trọng số
        y1f = y1.astype(np.float64)
        y2f = y2.astype(np.float64)

        kernel = np.ones((ksize, ksize), np.float64) / (ksize * ksize)

        mean1 = cv2.filter2D(y1f, -1, kernel, borderType=cv2.BORDER_REFLECT)
        var1 = cv2.filter2D(y1f ** 2, -1, kernel, borderType=cv2.BORDER_REFLECT) - mean1 ** 2
        var1 = np.maximum(var1, 0)

        mean2 = cv2.filter2D(y2f, -1, kernel, borderType=cv2.BORDER_REFLECT)
        var2 = cv2.filter2D(y2f ** 2, -1, kernel, borderType=cv2.BORDER_REFLECT) - mean2 ** 2
        var2 = np.maximum(var2, 0)

        total = var1 + var2 + 1e-10
        w1 = var1 / total
        w2 = var2 / total

        fused = w1 * c1.astype(np.float64) + w2 * c2.astype(np.float64)
        return np.clip(fused, 0, 255).astype(np.uint8)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _is_color(img: np.ndarray) -> bool:
        """Kiểm tra ảnh có phải color (3 channels) không."""
        return img.ndim == 3 and img.shape[2] == 3

    @staticmethod
    def _ensure_color(img: np.ndarray) -> np.ndarray:
        """Chuyển grayscale → BGR 3 channels nếu cần."""
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    @staticmethod
    def _match_size_color(
        img1: np.ndarray, img2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resize img2 cho khớp kích thước img1 (hỗ trợ cả color và gray)."""
        if img1.shape[:2] != img2.shape[:2]:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        return img1, img2
