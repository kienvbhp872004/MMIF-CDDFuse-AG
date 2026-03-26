"""
Demo & Test script cho 4 thuật toán decomposition fusion.
Test cả grayscale lẫn YCbCr color fusion.

Chạy: python algorithm/decomposition/demo_fusion.py
"""

import os
import sys
import time
import numpy as np
import cv2

# Thêm root project vào path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithm.decomposition import DWTFusion, NSSTFusion, PyramidFusion, GuidedFilterFusion


def create_test_images_gray(h: int = 256, w: int = 256):
    """Tạo 2 ảnh test grayscale."""
    img1 = np.zeros((h, w), dtype=np.uint8)
    for i in range(0, h, 20):
        img1[i:i + 10, :] = 200
    cv2.circle(img1, (w // 3, h // 3), 40, 255, -1)

    img2 = np.zeros((h, w), dtype=np.uint8)
    for j in range(0, w, 20):
        img2[:, j:j + 10] = 180
    cv2.rectangle(img2, (w // 2, h // 2), (w // 2 + 60, h // 2 + 60), 255, -1)

    return img1, img2


def create_test_images_color(h: int = 256, w: int = 256):
    """Tao 2 anh test COLOR (BGR)."""
    # Image 1: red-ish horizontal bands + blue circle
    img1 = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(0, h, 20):
        img1[i:i + 10, :] = [50, 80, 200]  # BGR: red-ish
    cv2.circle(img1, (w // 3, h // 3), 40, (255, 120, 50), -1)  # blue circle

    # Image 2: green-ish vertical bands + yellow rectangle
    img2 = np.zeros((h, w, 3), dtype=np.uint8)
    for j in range(0, w, 20):
        img2[:, j:j + 10] = [60, 180, 60]  # BGR: green-ish
    cv2.rectangle(img2, (w // 2, h // 2), (w // 2 + 60, h // 2 + 60),
                  (30, 220, 230), -1)  # yellow rect

    return img1, img2


def test_grayscale():
    """Test grayscale fusion."""
    print("\n" + "=" * 70)
    print("  GRAYSCALE FUSION (fuse)")
    print("=" * 70)

    img1, img2 = create_test_images_gray()
    print(f"\n  Test images: {img1.shape} (uint8, grayscale)")

    algorithms = [
        ("DWT",     DWTFusion(wavelet='db2', level=3)),
        ("NSST",    NSSTFusion(levels=2, directions=[4, 4])),
        ("Pyramid", PyramidFusion(levels=5, fusion_mode='max_abs')),
        ("Guided",  GuidedFilterFusion(radius=8, eps=0.01, n_scales=1)),
    ]

    for name, fuser in algorithms:
        t0 = time.perf_counter()
        fused = fuser.fuse(img1, img2)
        dt = time.perf_counter() - t0
        print(f"  {name:10s} → shape={fused.shape}, "
              f"range=[{fused.min()}, {fused.max()}], "
              f"time={dt * 1000:.1f}ms")


def test_ycbcr():
    """Test YCbCr color fusion."""
    print("\n" + "=" * 70)
    print("  YCbCr COLOR FUSION (fuse_ycbcr)")
    print("=" * 70)

    img1, img2 = create_test_images_color()
    print(f"\n  Test images: {img1.shape} (uint8, BGR color)")

    algorithms = [
        ("DWT",     DWTFusion(wavelet='db2', level=3)),
        ("NSST",    NSSTFusion(levels=2, directions=[4, 4])),
        ("Pyramid", PyramidFusion(levels=5, fusion_mode='max_abs')),
        ("Guided",  GuidedFilterFusion(radius=8, eps=0.01, n_scales=1)),
    ]

    for name, fuser in algorithms:
        t0 = time.perf_counter()
        fused = fuser.fuse_ycbcr(img1, img2)
        dt = time.perf_counter() - t0
        assert fused.ndim == 3 and fused.shape[2] == 3, \
            f"fuse_ycbcr should return 3-channel image, got {fused.shape}"
        print(f"  {name:10s} → shape={fused.shape}, "
              f"dtype={fused.dtype}, "
              f"time={dt * 1000:.1f}ms")


def test_auto_detect():
    """Test fuse_auto: auto-detect grayscale vs color."""
    print("\n" + "=" * 70)
    print("  AUTO-DETECT FUSION (fuse_auto)")
    print("=" * 70)

    img_gray1, img_gray2 = create_test_images_gray()
    img_color1, img_color2 = create_test_images_color()

    fuser = PyramidFusion(levels=5)

    # Auto gray
    t0 = time.perf_counter()
    fused_g = fuser.fuse_auto(img_gray1, img_gray2)
    dt = time.perf_counter() - t0
    assert fused_g.ndim == 2, f"Expected grayscale output, got {fused_g.shape}"
    print(f"\n  Grayscale input → output shape={fused_g.shape} (2D)")
    print(f"    time={dt * 1000:.1f}ms")

    # Auto color
    t0 = time.perf_counter()
    fused_c = fuser.fuse_auto(img_color1, img_color2)
    dt = time.perf_counter() - t0
    assert fused_c.ndim == 3, f"Expected color output, got {fused_c.shape}"
    print(f"  Color input     → output shape={fused_c.shape} (3-ch BGR)")
    print(f"    time={dt * 1000:.1f}ms")

    # Mixed: gray + color
    t0 = time.perf_counter()
    fused_m = fuser.fuse_auto(img_gray1, img_color2)
    dt = time.perf_counter() - t0
    assert fused_m.ndim == 3, f"Expected color output for mixed input, got {fused_m.shape}"
    print(f"  Mixed input     → output shape={fused_m.shape} (auto-convert gray→BGR)")
    print(f"    time={dt * 1000:.1f}ms")


def test_chroma_modes():
    """Test cac che do fusion cho chrominance."""
    print("\n" + "=" * 70)
    print("  CHROMINANCE FUSION MODES")
    print("=" * 70)

    img1, img2 = create_test_images_color()

    for mode in ['average', 'weighted', 'source1', 'source2']:
        fuser = DWTFusion(wavelet='db2', level=3)
        fuser.chroma_fusion = mode
        t0 = time.perf_counter()
        fused = fuser.fuse_ycbcr(img1, img2)
        dt = time.perf_counter() - t0
        print(f"  chroma_fusion='{mode:10s}' → shape={fused.shape}, "
              f"time={dt * 1000:.1f}ms")


if __name__ == '__main__':
    test_grayscale()
    test_ycbcr()
    test_auto_detect()
    test_chroma_modes()

    print("\n" + "=" * 70)
    print("  ALL TESTS PASSED!")
    print("=" * 70)
