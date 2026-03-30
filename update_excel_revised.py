import openpyxl

wb = openpyxl.load_workbook('Đỗ Trung Kiên_20224869.xlsx')
sheet = wb.active

# --- 1. Update Title and Field ---
sheet['A17'] = 'Cải tiến các phương pháp trộn ảnh y tế đa phương thức nhằm nâng cao chất lượng chẩn đoán y khoa.'
sheet['A18'] = 'Xử lý ảnh, Thị giác máy tính, Trí tuệ nhân tạo.'

# --- 2. Update Background and Objectives (3.5 Section) ---
# I need to find where 3.5 begins and rewrite those cells.
# Based on inspection, A38 was about 3.5.
sheet['A38'] = '3.5. Vấn đề thực tiễn đồ án giải quyết:'
sheet['A39'] = (
    "1. Bối cảnh:\n"
    "+ Trong chẩn đoán y tế hiện đại, các phương thức chụp ảnh khác nhau cung cấp thông tin đặc trưng (MRI cho thấy giải phẫu mềm, CT cho thấy cấu trúc xương, PET cung cấp thông tin chuyển đổi sinh hóa). \n"
    "+ Việc quan sát lẻ tẻ từng tấm ảnh gây khó khăn cho bác sĩ trong việc chẩn đoán chính xác vị trí bệnh lý. Trộn ảnh (Image Fusion) là kỹ thuật thiết yếu để tích hợp thông tin tối ưu từ nhiều nguồn ảnh vào một ảnh duy nhất."
)

sheet['A40'] = (
    "2. Mục tiêu và vấn đề giải quyết:\n"
    "+ Khắc phục các hạn chế còn tồn tại của các phương pháp trộn ảnh hiện nay như hiện tượng bóng ma (ghosting), mất độ tương phản và sai lệch màu sắc.\n"
    "+ Đề xuất phương pháp cải tiến dựa trên các kỹ thuật phân tách đa thang đo (Multiscale Decomposition) kết hợp với các luật trộn tối ưu (Fusion Rules) nhằm bảo toàn tối đa cả thông tin cấu trúc và thông tin chức năng.\n"
    "+ Xây dựng một quy trình đánh giá khách quan, định lượng và tin cậy cho bài toán trộn ảnh y tế."
)

# --- 3. Update Implementation Plan (Nội dung chi tiết) ---
# Nội dung 1 (1-4): Tìm hiểu tổng quan
sheet['A44'] = 'Nội dung 1: Khảo sát lý thuyết và phân tích yêu cầu hệ thống'
sheet['A46'] = (
    "Tuần 1-2: \n"
    "    + Nghiên cứu các ứng dụng của Image Fusion trong y khoa và các thách thức về chất lượng ảnh.\n"
    "    + Thu thập và chuẩn hóa bộ dữ liệu Harvard Medical Image. \n"
    "Tuần 3-4:\n"
    "    + Tìm hiểu các phép biến đổi miền tần số (DWT, NSST) và thuật toán lọc có hướng dẫn (Guided Filter).\n"
    "    + Xây dựng kiến trúc hệ thống tổng quát cho bài toán trộn ảnh."
)

# Nội dung 2 (5-7): Cài đặt cơ sở
sheet['A49'] = 'Nội dung 2: Xây dựng các mô hình nền tảng và bộ độ đo đánh giá'
sheet['A51'] = (
    "Tuần 5-7: \n"
    "- Cài đặt các thuật toán Baseline: DWT (Wavelet), NSST (Non-subsampled Shearlet), và Pyramid-based fusion.\n"
    "- Phát triển hệ thống tính toán 10+ độ đo chất lượng: SSIM, PSNR, VIF, Entropy, AG, UIQI...\n"
    "- Thực hiện các thử nghiệm sơ bộ để đánh giá ưu nhược điểm từng phương pháp cơ sở."
)

# Nội dung 3 (8-11): CẢI TIẾN (Nhiệm vụ trọng tâm)
sheet['A54'] = 'Nội dung 3: Đề xuất và triển khai giải pháp cải tiến Hybrid Fusion'
sheet['A56'] = (
    "Tuần 8-9: \n"
    "- Nghiên cứu cải tiến luật trộn dựa trên phân tích tương quan giữa các kênh (Cb, Cr) và độ sáng (Y).\n"
    "- Đề xuất tích hợp các phép lọc cạnh (Edge-preserving filters) để giảm hiện tượng lóa mờ.\n"
    "\n"
    "Tuần 10-11:\n"
    "- Triển khai thuật toán Hybrid kết hợp giữa NSST và Guided Filter hoặc tối ưu hóa tham số luật trộn.\n"
    "- Chứng minh tính hiệu quả của cải tiến thông qua so sánh trực tiếp với các phương pháp Baseline."
)

# Nội dung 4 (12-14): Thử nghiệm diện rộng và tối ưu
sheet['A59'] = 'Nội dung 4: Thử nghiệm thực tế và Đánh giá thống kê'
sheet['A61'] = (
    "Tuần 12-13: \n"
    "- Chạy thử nghiệm mù trên 50+ cặp ảnh bệnh lý (U não, đột quỵ, viêm nhiễm).\n"
    "- Thu thập dữ liệu, vẽ biểu đồ phân phối chất lượng và phân tích độ tin cậy.\n"
    "\n"
    "Tuần 14:\n"
    "- Tối ưu hóa hiệu năng tính toán, xử lý các trường hợp ảnh bị nhiễu nặng hoặc sai lệch định dạng chuẩn."
)

# Nội dung 5 (15-17): Hoàn thiện
sheet['A64'] = 'Nội dung 5: Đóng gói hệ thống và viết báo cáo đồ án'
sheet['A66'] = (
    "Tuần 15: \n"
    "- Hoàn thiện mã nguồn, viết hướng dẫn sử dụng và đóng gói sản phẩm.\n"
    "- Tổng hợp toàn bộ số liệu và viết thảo luận về kết quả đạt được.\n"
    "Tuần 16-17:\n"
    "- Hoàn thiện báo cáo ĐATN bản cứng, chuẩn bị Slide và kịch bản bảo vệ đồ án."
)

# --- 4. Final Save ---
new_filename = 'Đỗ Trung Kiên_20224869_v2.xlsx'
wb.save(new_filename)
print(f"Successfully saved revised plan as {new_filename}")
