"""
Decomposition-based Image Fusion Algorithms
============================================

Module chứa các thuật toán phân tách & tái tạo ảnh phục vụ fusion:

1. DWT   - Discrete Wavelet Transform (multi-level)
2. NSST  - Non-subsampled Shearlet Transform
3. Pyramid - Gaussian / Laplacian Pyramid
4. Guided Filter & Edge-Preserving Decomposition

Tất cả đều hỗ trợ fusion trên không gian màu YCbCr qua BaseFusion:
- fuse(img1, img2)          → fusion grayscale
- fuse_ycbcr(img1, img2)    → fusion trong YCbCr (Y: decomposition, Cb/Cr: weighted)
- fuse_auto(img1, img2)     → auto-detect grayscale hay color
"""

from ._base import BaseFusion
from .dwt_fusion import DWTFusion
from .nsst_fusion import NSSTFusion
from .pyramid_fusion import PyramidFusion
from .guided_filter_fusion import GuidedFilterFusion

__all__ = [
    "BaseFusion",
    "DWTFusion",
    "NSSTFusion",
    "PyramidFusion",
    "GuidedFilterFusion",
]
