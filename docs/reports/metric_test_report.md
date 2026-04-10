# BÁO CÁO TỔNG HỢP 22 METRIC ĐÁNH GIÁ ẢNH HỢP NHẤT

Dựa trên tài liệu Liu 2012 và các yêu cầu bổ sung. Thuật toán sử dụng: **Wavelet Transform**.

## 1. Ma trận Kiểm thử (Lượt trích 5x5)

#### Ảnh A (Nguồn 1):

```
[[101 142  64 121 110]
 [ 91 141 109 129  64]
 [ 57  96  84 127 130]
 [129 131 102  73  75]
 [ 61  83  82  97  72]]
```

#### Ảnh B (Nguồn 2):

```
[[161 161 193 194 123]
 [143 104 169 125 167]
 [101 183 164 162 172]
 [192 171 110 113 159]
 [116 196 165 177 152]]
```

#### Ảnh F (Hợp nhất theo Wavelet):

```
[[116.75  181.75  130.75  191.75  138.875]
 [ 98.75  124.75  106.75  122.75   99.375]
 [ 64.75  146.75  121.125 164.125 174.5  ]
 [169.75  148.75  105.625  76.625 119.5  ]
 [ 48.875 128.875 121.5   136.5   102.5  ]]
```

## 2. Kết quả chi tiết (22 Metrics)

| STT | Tên Chỉ số (Metric)      | Kết quả         | Miền giá trị (Range) | Phân nhóm    |
| :-- | :----------------------- | :-------------- | :------------------- | :----------- |
| 1   | Cường độ sáng TB (Mean)  | **124.131348**  | [0, 255]             | Thống kê     |
| 2   | Độ tương phản (Variance) | **1128.638693** | [0, inf)             | Thống kê     |
| 3   | Độ sắc nét (Avg Grad)    | **24.103971**   | [0, inf)             | Thống kê     |
| 4   | Entropy                  | **6.968745**    | [0, 8]               | Thống kê     |
| 5   | PSNR                     | **47.481169**   | [0, inf) dB          | Truyền thống |
| 6   | SSIM (Standard)          | **1.231948**    | [0, 2]               | Truyền thống |
| 7   | RMSE                     | **1.161351**    | [0, inf)             | Truyền thống |
| 8   | Mutual Information (MI)  | **5.812966**    | [0, inf)             | Truyền thống |
| 9   | Cross Entropy (CE)       | **0.679475**    | [0, inf)             | Truyền thống |
| 10  | Spatial Frequency (SF)   | **69.514453**   | [0, inf)             | Truyền thống |
| 11  | Edge Intensity (EI)      | **139.028307**  | [0, inf)             | Truyền thống |
| 12  | Q_G (Petrovic)           | **0.326703**    | [0, 1]               | Liu 2012     |
| 13  | Q_CB (Chen-Blum)         | **0.999359**    | [0, 1]               | Liu 2012     |
| 14  | Q_CV (Chen-Varshney)     | **480.871817**  | [0, inf)             | Liu 2012     |
| 15  | Q_M (Wavelet QM)         | **0.022635**    | [0, 1]               | Liu 2012     |
| 16  | Q_C (Piella Structural)  | **0.619338**    | [-1, 1]              | Liu 2012     |
| 17  | Q_S (Piella Saliency)    | **0.608527**    | [-1, 1]              | Liu 2012     |
| 18  | Q_Y (Yang SSIM)          | **0.690964**    | [0, 1]               | Liu 2012     |
| 19  | NMI (Normalized MI)      | **1.082649**    | [0, 2]               | Liu 2012     |
| 20  | Q_SF (Relative SF)       | **0.050201**    | [0, 1]               | Liu 2012     |
| 21  | Q_NCIE (NCC Entropy)     | **0.834390**    | [0, 1]               | Nâng cao     |
| 22  | Q_TE (Tsallis Entropy)   | **-3.270922**   | [0, inf)             | Nâng cao     |
