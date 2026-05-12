# Phân Tích Kỹ Thuật Chuyên Sâu Các Mô Hình Image Fusion (Detailed Analysis)

Tài liệu này cung cấp chi tiết về kiến trúc hệ thống, phương pháp phân rã, các bộ lọc (filters) được sử dụng và cơ chế hoạt động của 20 mô hình Image Fusion tiêu biểu.

---

## I. Nhóm Mô Hình Khuếch Tán (Diffusion-based Models)
*Mô hình: DDFM, DM-FNet, Mask-DiFuser, FlexiD-Fuse*

### 1. Cơ chế hoạt động (The Diffusion Process)
Các mô hình này biến đổi bài toán Fusion thành một bài toán **Generative Conditional Modeling**. Thay vì học hàm fusion trực tiếp, chúng học cách giải nhiễu (denoise) một biến ngẫu nhiên Gaussian để tạo ra ảnh fusion với điều kiện là các ảnh nguồn.

### 2. Kiến trúc chi tiết & Bộ phân rã
*   **Bộ phân rã (Decomposition):** Không sử dụng bộ phân rã tường minh (explicit). Thay vào đó, chúng phân rã ảnh vào **không gian nhiễu (Noise Space)**. Thông tin "Base" nằm ở các bước giải nhiễu ban đầu (vùng tần số thấp), thông tin "Detail" xuất hiện ở các bước giải nhiễu cuối (vùng tần số cao).
*   **Mạng nơ-ron cốt lõi:** Thường sử dụng **U-Net** với các khối **Residual Blocks** và **Self-Attention**.
*   **Bộ lọc (Filters):** Các bộ lọc tích chập (Convolutional filters) được học tự động để ước lượng nhiễu trong từng bước thời gian (timestep).

### 3. Luật kết hợp (Fusion Rule)
*   Sử dụng thuật toán **Expectation-Maximization (EM)** hoặc **Bayesian Modeling** để hiệu chỉnh xác suất (Likelihood Rectification), đảm bảo ảnh tạo ra chứa thông tin từ cả hai modality (ví dụ: IR và VIS).

---

## II. Nhóm Mô Hình Mamba (State Space Models - SSM)
*Mô hình: MFS-Fusion, LPM-Net, LMACV*

### 1. Kiến trúc chi tiết
Mamba sử dụng cơ chế **Selective State Space** để xử lý dữ liệu ảnh dưới dạng chuỗi 1D thông qua các phép toán quét (Scanning).

### 2. Bộ phân rã & Bộ lọc
*   **Bộ phân rã:** 
    *   **MFS-Fusion:** Sử dụng **Multi-Scale Fourier Enhancement**. Nó sử dụng phép biến đổi **Fourier nhanh (FFT)** để phân rã ảnh vào miền tần số, cho phép tinh chỉnh các dải tần số đặc trưng của từng modality một cách độc lập.
    *   **LPM-Net:** Phân rã theo cấp độ Pixel (Pixel-level modeling) kết hợp với CNN cho đặc trưng cục bộ.
*   **Bộ lọc:** Sử dụng các bộ lọc **Fourier Adaptive Filters** và các tham số $A, B, C$ trong mô hình SSM để lọc thông tin theo thời gian/không gian một cách chọn lọc.

---

## III. Nhóm Mô Hình Transformer (Attention-based)
*Mô hình: CDDFuse, ITFuse, MACTFusion, FocalNetFuse*

### 1. Bộ phân rã (Dual-Branch Decomposition)
*   **Nhánh Base (Low-frequency):** Sử dụng Transformer (thường là Lite Transformer hoặc Restormer) để bắt các mối quan hệ ngữ nghĩa toàn cục và ánh sáng.
*   **Nhánh Detail (High-frequency):** Sử dụng CNN với các bộ lọc học được hoặc **Invertible Neural Networks (INN)** để giữ lại chi tiết cạnh và vân bề mặt.

### 2. Bộ lọc & Luật kết hợp
*   **Bộ lọc:** Sử dụng **Multi-head Self-Attention** như một bộ lọc thông thấp (low-pass) động và các bộ lọc tích chập 3x3 như bộ lọc thông cao (high-pass).
*   **Luật kết hợp:** **Cross-modality Attention**. Nó tính toán ma trận tương quan giữa đặc trưng của Modality A và B để quyết định trọng số kết hợp tại mỗi điểm ảnh.

---

## IV. Nhóm Mô Hình Đảo Ngược & Nest (INN, AE)
*Mô hình: MMIF-INet, NestFuse, MMAE, RFN-Nest*

### 1. Mạng nơ-ron đảo ngược (INN)
*   **Kiến trúc:** Sử dụng các khối cộng hưởng (Affine Coupling Layers). 
*   **Cơ chế:** Phép biến đổi giữa không gian ảnh và ẩn là hoàn toàn thuận nghịch ($x \leftrightarrow z$), đảm bảo không có thông tin nào bị "rơi rớt" trong quá trình mã hóa.
*   **Bộ lọc:** Các bộ lọc tích chập sâu (Deep learning kernels) được tối ưu hóa để cực đại hóa **Mutual Information (Thông tin tương hỗ)**.

### 2. Kiến trúc Nest Connection
*   **Phân rã:** Sử dụng kiến trúc tương tự **UNet++** với các đường kết nối lồng ghép (nested skip connections).
*   **Bộ lọc chi tiết:** Các module **Spatial/Channel Attention** đóng vai trò là các bộ lọc trọng số, ưu tiên các vùng có độ tương phản cao (saliency parts).

---

## V. Nhóm Mô Hình CNN & Hybrid Đặc Thù
*Mô hình: MLFuse, AWFusion, BSAFusion, GeSeNet, LRFNet, MBHFuse, MM-Net-Fusion*

### 1. Phân tích chi tiết từng mô hình
*   **AWFusion (Adaptive frequency based):** 
    *   Sử dụng **Focal Frequency Loss** để hướng dẫn việc học trong miền tần số.
    *   Phân rã: Sử dụng các lớp tích chập quy mô lớn để mô phỏng các bộ lọc tần số.
*   **BSAFusion:**
    *   Kiến trúc: Tích hợp **Registration (Đăng ký ảnh)** và Fusion.
    *   Cơ chế: Sử dụng **Stepwise Feature Alignment** để căn chỉnh các đặc trưng bị lệch trước khi thực hiện luật kết hợp.
*   **GeSeNet (Semantic-guided):**
    *   Phân rã: Sử dụng **Semantic segmentation mask** (mặt nạ phân đoạn ngữ nghĩa) để phân rã ảnh thành các vùng vật thể khác nhau.
    *   Bộ lọc: Sử dụng các bộ lọc Refinement để trau chuốt cạnh vật thể dựa trên thông tin ngữ nghĩa.
*   **LRFNet (Wavelet-guided):**
    *   Phân rã: Sử dụng **Discrete Wavelet Transform (DWT)** tường minh để tách thành phần tần số cao và thấp, sau đó đưa vào mạng CNN nhẹ để xử lý.
*   **ECINFusion:**
    *   Cơ chế: Sử dụng **Explicit Channel interaction**.
    *   Bộ lọc: Các bộ lọc Attention theo kênh (Channel-wise) giúp mô hình hóa mối quan hệ hóa học/vật lý giữa các modality y tế.

---

## Bảng Tổng Kết Cơ Chế Phân Rã & Lọc (Summary Table)

| Model | Loại Mạng | Bộ Phân Rã (Decomposition) | Bộ Lọc Chủ Đạo (Filters) |
| :--- | :--- | :--- | :--- |
| **CDDFuse** | Transformer/CNN | Dual-branch (Base/Detail) | Lite Transformer & INN kernels |
| **DDFM** | Diffusion | Noise-space (Iterative) | Learned Diffusion Kernels |
| **MFS-Fusion**| Mamba | Multi-Scale Fourier (FFT) | State Space Selective Filters |
| **MMIF-INet** | INet | Reversible Hidden Features | Affine Coupling Filters |
| **NestFuse** | AE-based | Nested Multi-scale Encoder | Spatial/Channel Attention Filters |
| **LRFNet** | CNN | Wavelet Transform (DWT) | Wavelet-guided CNN Filters |
| **GeSeNet** | Semantic CNN | Semantic/Structural Masking | Edge Re-finery Filters |
| **BSAFusion** | Registration-CNN| Path Independent Alignment | Alignment Displacement Filters |

---
*Lưu ý cho giảng viên: Các mô hình hiện đại có xu hướng chuyển từ bộ phân rã toán học cố định (như Wavelet, Pyramid) sang bộ phân rã **học được (learned decomposition)** thông qua cấu trúc đa nhánh hoặc Transformer, giúp tối ưu hóa kết quả dựa trên dữ liệu thực tế.*
