# Comprehensive Metrics Unit Test Report

Generated at: 2026-04-08T03:34:49

## 1. Small Sample Matrices (5x5)

These values were used for simple metrics (IE, AG, SF, etc.):

### Matrix A (5x5 Sample - top left 3x3)

```
[[102. 179.  92.]
 [ 71. 188.  20.]
 [210. 214.  74.]]
```

### Matrix B (5x5 Sample - top left 3x3)

```
[[157.  37. 129.]
 [ 20. 160. 203.]
 [252. 235.  88.]]
```

### Fused Matrix F (Average - top left 3x3)

```
[[129.5 108.  110.5]
 [ 45.5 174.  111.5]
 [231.  224.5  81. ]]
```

## 2. Unit Test Results

| Metric Name       | Matrix Size | Result Value | Category    |
| :---------------- | :---------- | :----------- | :---------- |
| IE (Entropy)      | 5x5         | 4.48385619   | Information |
| Avg Gradient      | 5x5         | 70.79559590  | Visual      |
| SF (Spatial Freq) | 5x5         | 95.69247619  | Visual      |
| Var (Std Dev)     | 5x5         | 54.32466843  | Visual      |
| Mutual Info       | 5x5         | 6.10504100   | Information |
| Cross Entropy     | 5x5         | 0.06000000   | Information |
| PSNR              | 5x5         | 38.19010551  | Comparative |
| RMSE              | 5x5         | 9.86438037   | Comparative |
| SSIM              | 15x15       | 1.49392736   | Comparative |
| FMI               | 15x15       | 0.43358704   | Feature     |
| QG (Qabf)         | 15x15       | 0.43997449   | Edge        |
| NCIE              | 15x15       | 0.90082955   | Information |
| NABF (Synthetic)  | 15x15       | 0.00000000   | Artifact    |
| Piella QC         | 15x15       | 0.80286640   | Structural  |
| Piella QS         | 15x15       | 0.76309102   | Structural  |

## 3. Real Image Verification (NABF)

Since NABF is often 0 on random noise, we test it on a real medical image pair (SPECT-MRI):

- **NABF (Real Image 25015.png)**: `0.15567219`
