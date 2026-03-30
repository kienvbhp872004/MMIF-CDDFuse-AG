# Framework for Multi-Modality Image Fusion & Quantitative Evaluation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Integrated-ee4c2c.svg)](https://pytorch.org/)

Hệ thống xử lý và đánh giá hợp nhất ảnh đa phương thức với trọng tâm là ứng dụng trong chẩn đoán hình ảnh y tế. Dự án cung cấp giải pháp toàn diện từ các phương pháp phân rã toán học truyền thống đến các mô hình học sâu (SOTA Deep Learning).

---

## Giới thiệu dự án

Hợp nhất ảnh y tế (Medical Image Fusion) đóng vai trò then chốt trong việc hỗ trợ các bác sĩ chẩn đoán bệnh chính xác hơn bằng cách kết hợp thông tin từ nhiều nguồn ảnh khác nhau (như MRI, PET, SPECT, CT).

Dự án này được xây dựng nhằm giải quyết bài toán tối ưu hóa việc trích xuất và kết hợp các đặc trưng cấu trúc (từ MRI/CT) và đặc trưng chức năng (từ PET/SPECT). Hệ thống không chỉ cung cấp các kết quả hợp nhất trực quan mà còn đi kèm bộ công cụ đánh giá định lượng nghiêm ngặt dựa trên hơn 22 chỉ số toán học chuẩn quốc tế.

## Mục tiêu dự án

Dự án tập trung vào các mục tiêu nghiên cứu và phát triển chính sau:

1.  **Nghiên cứu các phương pháp phân rã đa tỷ lệ (Multiscale Decomposition):** Triển khai và tối ưu hóa các bộ lọc toán học như NSST, DWT, Guided Filter nhằm tách lọc chi tiết ảnh hiệu quả.
2.  **So sánh hiệu năng giữa Truyền thống và Deep Learning:** Xây dựng cơ sở so sánh khách quan giữa các phương pháp học sâu hiện đại (CDDFuse, DAF-Net) và các phương pháp toán học cổ điển.
3.  **Hỗ trợ đa dạng Modality:** Thiết lập pipeline xử lý ổn định cho nhiều cặp thuật toán (MRI-PET, MRI-SPECT, CT-MRI) cả ở định dạng Grayscale và Color (YCbCr conversion).
4.  **Tự động hóa quy trình đánh giá:** Loại bỏ cảm tính trong việc đánh giá kết quả bằng cách tự động hóa quá trình tính toán metric và xuất báo cáo khoa học.

---

## Các tính năng nổi bật

- **Đa dạng phương pháp phân rã:** Triển khai 4 kỹ thuật cốt lõi: **NSST** (Non-subsampled Shearlet Transform), **DWT**, **Laplacian Pyramid**, và **Guided Filter Fusion**.
- **Tích hợp State-of-the-art Models:** Hỗ trợ trực tiếp các mô hình học sâu tiên tiến: **CDDFuse**, **DAF-Net**, và **PSFusion**.
- **Xử lý màu thông minh:** Tự động chuyển đổi và xử lý trên không gian màu YCbCr giúp bảo toàn độ tương phản và độ tự nhiên của màu sắc.
- **Hệ thống Metric toàn diện:** Đánh giá dựa trên 22+ chỉ số (Entropy, SSIM, PSNR, Mutual Information, Average Gradient, Edge Preservation...), hỗ trợ xuất báo cáo định dạng Excel và CSV.
- **Thiết kế hướng đối tượng (OOP):** Cấu trúc mã nguồn module hóa giúp dễ dàng bảo trì và mở rộng thêm các thuật toán mới.

---

## Công nghệ sử dụng

- **Ngôn ngữ lập trình:** Python 3.8+
- **Thư viện xử lý ảnh:** OpenCV, PyWavelets, Scikit-Image, Scipy.
- **Deep Learning Framework:** PyTorch.
- **Phân tích & Thống kê:** Pandas, Matplotlib, Seaborn, openpyxl.
- **Dataset:** Harvard Medical Image Fusion Datasets.

---

## Cấu trúc mã nguồn

```text
Image-Fusion
├── algorithm/          # Triển khai lõi các thuật toán fusion
│   ├── decomposition/  # DWT, NSST, Pyramid, Guided Filter
│   └── _base.py        # Base Class cho thiết kế Modular
├── metric/             # Bộ sưu tập các hàm tính toán 22+ metrics
├── models/             # Chứa pretrained weights của các mô hình Deep Learning
├── data/               # Dữ liệu ảnh đầu vào (MRI, PET, SPECT, CT)
├── evaluate_decomposition.py # Script đánh giá so sánh các thuật toán phân rã
├── visualization.py       # Công cụ biểu đồ và heatmap kết quả
├── main.py                # Điểm khởi đầu demo nhanh
└── requirements.txt       # Danh sách thư viện phụ thuộc
```

---

## Kết quả và Đánh giá

Hệ thống cho phép đánh giá trực quan và định lượng để xác định tham số tối ưu cho từng loại dữ liệu ảnh:

| Thuật toán        | PSNR (dB) | SSIM | Entropy | Mutual Info | Time (s) |
| :---------------- | :-------: | :--: | :-----: | :---------: | :------: |
| **NSST**          |   48.25   | 1.15 |  7.12   |    6.02     |   1.25   |
| **DWT**           |   47.41   | 1.23 |  6.96   |    5.81     |   0.45   |
| **Guided Filter** |   42.10   | 0.98 |  6.85   |    5.12     |   0.12   |

---

## Hướng dẫn cài đặt

1.  **Clone repository:**

    ```bash
    git clone https://github.com/kienvbhp872004/Image-Fusion.git
    cd Image-Fusion
    ```

2.  **Cài đặt môi trường:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Thực thi demo:**

    ```bash
    python main.py
    ```

4.  **Chạy quy trình đánh giá:**
    ```bash
    python evaluate_decomposition.py
    ```

---

## Thông tin liên hệ

- **Tác giả:** Đỗ Trung Kiên
- **Đơn vị:** Đại học Bách Khoa Hà Nội (HUST)
- **Mã số sinh viên:** 20224869
- **Email:** kien.dt224869@sis.hust.edu.vn
- **Chuyên môn:** Computer Vision, Medical Imaging.

---

> [!IMPORTANT]
> Đây là dự án nghiên cứu trong khuôn khổ học tập tại HUST. Mọi chi tiết về mã nguồn và hợp tác nghiên cứu vui lòng liên hệ trực tiếp.
