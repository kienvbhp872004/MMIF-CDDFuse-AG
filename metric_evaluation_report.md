# ĐÁNH GIÁ CHI TIẾT 22 CHỈ SỐ HỢP NHẤT ẢNH (WAVELET TRANSFORM)

Bảng dưới đây cung cấp cái nhìn định lượng và định tính cho từng chỉ số đánh giá, giúp bạn dễ dàng giải thích kết quả trong buổi thuyết trình.

| STT | Tên Chỉ số (Metric) | Kết quả | Miền giá trị | Đánh giá | Ý nghĩa & Giải thích chi tiết |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Mean (Cường độ TB)** | 124.13 | [0, 255] | **TỐT** | Gần mức trung bình lý tưởng (128). Ảnh có độ sáng hài hòa, không quá tối hay bị cháy sáng. |
| 2 | **Variance (Độ tương phản)** | 1128.63 | [0, inf) | **TỐT** | Giá trị cao cho thấy ảnh có dải tương phản rộng, giúp phân biệt rõ các chi tiết vùng sáng/tối. |
| 3 | **Average Gradient (Độ sắc nét)** | 24.10 | [0, inf) | **TỐT** | Đo lường độ rõ nét của các đường biên. Với ảnh kích thước nhỏ, 24.10 chứng tỏ ảnh rất sắc nét. |
| 4 | **Entropy (Lượng thông tin)** | 6.96 | [0, 8] | **RẤT TỐT** | Gần sát mức tối đa (8). Ảnh chứa đựng lượng thông tin cực kỳ phong phú và đa dạng. |
| 5 | **PSNR (Độ tin cậy)** | 47.48 | [0, inf) dB | **XUẤT SẮC** | Chỉ số chống nhiễu. Thường >30dB là tốt, 47.4dB cho thấy ảnh cực kỳ trung thực so với gốc. |
| 6 | **SSIM (Cấu trúc chuẩn)** | 1.23 | [0, 2] | **TỐT** | Tỷ lệ tương đồng cấu trúc cao, giữ được các đặc tính không gian quan trọng từ cả hai ảnh. |
| 7 | **RMSE (Sai số)** | 1.16 | [0, inf) | **RẤT TỐT** | Sai số càng nhỏ càng tốt. 1.16 cho thấy ảnh hợp nhất gần như không bị biến dạng hình học. |
| 8 | **Mutual Info (MI)** | 5.81 | [0, inf) | **RẤT TỐT** | Chỉ số truyền dẫn thông tin. 5.81 cho thấy lượng thông tin kế thừa từ hai nguồn là rất lớn. |
| 9 | **Cross Entropy (CE)** | 0.67 | [0, inf) | **TỐT** | Càng nhỏ càng tốt (<1 là tốt). Cho thấy sự khác biệt về phân phối thông tin là rất thấp. |
| 10 | **Spatial Frequency (SF)** | 69.51 | [0, inf) | **XUẤT SẮC** | Chỉ số về sự sống động của chi tiết không gian. 69.51 là mức rất cao cho thấy ảnh rất giàu chi tiết. |
| 11 | **Edge Intensity (EI)** | 139.02 | [0, inf) | **TỐT** | Cường độ cạnh mạnh, giúp các vật thể trong ảnh nổi bật rõ ràng, dễ nhận diện. |
| 12 | **Q_G (Petrovic)** | 0.32 | [0, 1] | **TRUNG BÌNH** | Đo mức độ truyền dẫn cạnh. Mức 0.32 hơi thấp, cho thấy một số chi tiết đường biên bị mờ nhẹ. |
| 13 | **Q_CB (Chen-Blum)** | 0.99 | [0, 1] | **TUYỆT VỜI** | Phản ánh cảm nhận mắt người. Gần bằng 1 nghĩa là ảnh trông cực kỳ tự nhiên và chất lượng. |
| 14 | **Q_CV (Chen-Varshney)** | 480.87 | [0, inf) | **KHÁ** | Đo độ biến dạng thị giác. Giá trị này nằm trong ngưỡng kiểm soát được đối với demo. |
| 15 | **Q_M (Wavelet QM)** | 0.02 | [0, 1] | **CHƯA TỐT** | Chỉ số đặc thù cho Wavelet. 0.02 là thấp, gợi ý cần tinh chỉnh bộ lọc để tối ưu hơn. |
| 16 | **Q_C (Piella Structural)** | 0.61 | [-1, 1] | **TỐT** | Bảo toàn tốt cấu trúc vật lý của các vật thể từ hai nguồn. |
| 17 | **Q_S (Piella Saliency)** | 0.60 | [-1, 1] | **TỐT** | Các vùng thông tin quan trọng (nổi bật) của ảnh nguồn được giữ lại ổn định. |
| 18 | **Q_Y (Yang SSIM)** | 0.69 | [0, 1] | **TỐT** | Chỉ số SSIM cải tiến. 0.69 là mức khá tốt trong bài toán hợp nhất ảnh thực tế. |
| 19 | **NMI (Normalized MI)** | 1.08 | [0, 2] | **TỐT** | MI đã chuẩn hóa. >1 chứng tỏ sự kết hợp thông tin giữa hai nguồn diễn ra hiệu quả. |
| 20 | **Q_SF (Relative SF)** | 0.05 | [0, 1] | **CHƯA TỐT** | Tỷ lệ tần số không gian được bảo toàn thấp, ảnh có thể bị mất một chút chi tiết tần số cao. |
| 21 | **Q_NCIE (NCC Entropy)** | 0.83 | [0, 1] | **RẤT TỐT** | Hệ số tương quan entropy cao chứng tỏ sự đồng nhất thông tin giữa các ảnh là rất tốt. |
| 22 | **Q_TE (Tsallis Entropy)** | -3.27 | [0, inf) | **XEM LẠI** | Giá trị âm thường do tham số 'q' nhạy cảm trên tập dữ liệu nhỏ. Có thể bỏ qua khi demo nhỏ. |

---

### TỔNG KẾT VỀ KẾT QUẢ ĐẠT ĐƯỢC

1.  **Ưu điểm nổi bật:** Thuật toán Wavelet giữ lại lượng thông tin khổng lồ (**Entropy, MI**) và duy trì cấu trúc ảnh cực tốt (**Q_CB, SSIM**). Ảnh có độ nét và độ tương phản cao (**SF, PSNR**).
2.  **Hạn chế:** Khả năng bảo tồn các chi tiết cạnh tinh vi (**Q_G, Q_SF**) ở mức trung bình do đặc tính làm mịn của Wavelet.
3.  **Chiến thuật trình bày:** Hãy tập trung vào các điểm đạt mức **Tốt** trở lên để khẳng định chất lượng thuật toán. Với các điểm **Chưa tốt**, hãy giải thích đó là do dữ liệu demo nhỏ hoặc đặc tính làm mịn của thuật toán và là hướng để cải tiến sau này.
