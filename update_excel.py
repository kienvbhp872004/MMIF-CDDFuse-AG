import openpyxl

wb = openpyxl.load_workbook('Nguyễn Trung Long_20224874.xlsx')
sheet = wb.active

# 1. Update Student Info
sheet['C9'] = 'Đỗ Trung Kiên'
sheet['I9'] = 20224869
sheet['C10'] = '' # Clear old phone number
sheet['C11'] = 'kien.dt224869@sis.hust.edu.vn'
sheet['C13'] = 'Phạm Đăng Hải'

# 2. Update Topic
sheet['A17'] = 'Nghiên cứu và phát triển hệ thống trộn ảnh y tế đa nguồn sử dụng các phương pháp phân tách ảnh'
sheet['A18'] = 'Xử lý ảnh, Trí tuệ nhân tạo'

# 3. Update Plan (Nội dung thực hiện)
sheet['A44'] = 'Nội dung 1: Tìm hiểu tổng quan về bài toán trộn ảnh y tế và các kỹ thuật phân tách'
sheet['A46'] = (
    "Tuần 1 + 2: \n"
    "    + Khảo sát tài liệu về Image Fusion, các hướng tiếp cận dựa trên thực thể và miền tần số.\n"
    "    + Tìm hiểu các bộ dữ liệu y tế công khai (MRI, PET, CT).\n"
    "Tuần 3 + 4:\n"
    "    + Nghiên cứu lý thuyết các thuật toán DWT, NSST, Pyramid và Guided Filter."
)

sheet['A49'] = 'Nội dung 2: Xây dựng và cài đặt các thuật toán trộn ảnh'
sheet['A51'] = (
    "- Cài đặt các phương pháp trộn ảnh dựa trên phân tách: DWT, NSST, Laplacian Pyramid.\n"
    "- Thực hiện trộn ảnh trên các kênh màu YCbCr cho ảnh màu."
)

sheet['A54'] = 'Nội dung 3: Xây dựng hệ thống đánh giá chất lượng ảnh'
sheet['A56'] = (
    "Tuần 6 + 7: \n"
    "- Nghiên cứu và cài đặt các độ đo: Entropy, SSIM, PSNR, VIF, AG.\n"
    "\n"
    "Tuần 8 + 9:\n"
    "- Xây dựng kịch bản thử nghiệm tự động trên tập dữ liệu lớn."
)

sheet['A59'] = 'Nội dung 4: Thử nghiệm và phân tích kết quả'
sheet['A61'] = (
    "Tuần 10 + 11: \n"
    "- Chạy thử nghiệm trên các cặp ảnh MRI-PET và CT-MRI.\n"
    "- Thu thập dữ liệu và vẽ biểu đồ so sánh hiệu năng giữa các thuật toán.\n"
    "\n"
    "Tuần 12 + 13: \n"
    "- Phân tích ưu nhược điểm của từng phương pháp trong các tình huống cụ thể."
)

sheet['A64'] = 'Nội dung 5: Đóng gói và hoàn thiện báo cáo'
sheet['A66'] = (
    "Tuần 14: \n"
    "- Tối ưu hóa mã nguồn, đóng gói công cụ hỗ trợ trộn ảnh.\n"
    "- Viết báo cáo chi tiết các kết quả đạt được.\n"
    "Tuần 15: \n"
    "- Hoàn thiện báo cáo đồ án, thiết kế slide và chuẩn bị bảo vệ."
)

# 4. Update signatures/footer
sheet['I75'] = 'Kiên'

# Save the new file
new_filename = 'Đỗ Trung Kiên_20224869.xlsx'
wb.save(new_filename)
print(f"Successfully saved as {new_filename}")
