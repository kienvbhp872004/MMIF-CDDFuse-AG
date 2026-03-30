import openpyxl

wb = openpyxl.load_workbook('Đỗ Trung Kiên_20224869.xlsx')
sheet = wb.active

# --- 1. Update Title and Field ---
sheet['A17'] = 'Cải tiến các phương pháp trộn ảnh y tế đa phương thức nhằm nâng cao chất lượng chẩn đoán y khoa.'
sheet['A18'] = 'Xử lý ảnh, Thị giác máy tính, Trí tuệ nhân tạo.'

# --- 2. Update Background and Objectives (Main Block) ---
# A39:K41 is merged, so we write only to A39.
sheet['A39'] = (
    "1. Bối cảnh đồ án:\n"
    "+ Trong chẩn đoán y học hiện đại, các nguồn ảnh y tế khác nhau (MRI, CT, PET, SPECT) cung cấp các thông tin bổ trợ quan trọng. Việc trộn ảnh (Image Fusion) giúp tích hợp các ưu điểm của từng nguồn ảnh vào một tấm ảnh duy nhất để cải thiện độ tin cậy của chẩn đoán.\n"
    "2. Vấn đề giải quyết & Mục tiêu cải tiến:\n"
    "+ Khắc phục hiện tượng bóng ma (ghosting) và mất độ tương phản trong các phương pháp truyền thống.\n"
    "+ Nghiên cứu cải tiến luật trộn Hybrid (Hybrid Fusion Rules) kết hợp giữa các kỹ thuật lọc bảo toàn cạnh (Edge-preserving filtering) và phân tách đa thang đo (Multiscale Decomposition).\n"
    "+ Xây dựng bộ công cụ đánh giá định lượng giúp kiểm chứng tính hiệu quả của phương pháp đề xuất."
)

# --- 3. Update Implementation Plan (Weekly) ---
sheet['A44'] = 'Nội dung 1: Khảo sát lý thuyết và phân tích yêu cầu bài toán'
sheet['A46'] = (
    "Tuần 1-2: Nghiên cứu các ứng dụng của Image Fusion trong y khoa và các thách thức về chất lượng ảnh. Thu thập dữ liệu Harvard.\n"
    "Tuần 3-4: Tìm hiểu lý thuyết các phép biến đổi miền tần số (DWT, NSST) và thuật toán lọc hướng dẫn (Guided Filter)."
)

sheet['A49'] = 'Nội dung 2: Xây dựng các thuật toán Baseline và bộ độ đo đánh giá'
sheet['A51'] = (
    "Tuần 5-7: Cài đặt các thuật toán cơ sở: DWT, NSST, Laplacian Pyramid.\n"
    "Phát triển hệ thống tính toán 10+ độ đo chất lượng (SSIM, PSNR, VIF, Entropy, AG, UIQI...)."
)

sheet['A54'] = 'Nội dung 3: Đề xuất và thực hiện giải pháp cải tiến Hybrid Fusion'
sheet['A56'] = (
    "Tuần 8-9: Nghiên cứu cải tiến luật trộn dựa trên phân tích tương quan kênh màu (Cb, Cr) và độ sáng (Y).\n"
    "Tuần 10-11: Triển khai thuật toán Hybrid kết hợp NSST và Guided Filter để bảo toàn tối đa cả cạnh và kết cấu ảnh."
)

sheet['A59'] = 'Nội dung 4: Thử nghiệm thực tế và phân tích kết quả định lượng'
sheet['A61'] = (
    "Tuần 12-13: Chạy thử nghiệm trên tập dữ liệu lớn các cặp ảnh bệnh lý. Thu thập dữ liệu và vẽ biểu đồ so sánh hiệu năng.\n"
    "Tuần 14: Tối ưu hóa tham số thuật toán và phân tích các trường hợp sai số để rút ra kết luận."
)

sheet['A64'] = 'Nội dung 5: Đóng gói sản phẩm và hoàn thiện báo cáo Đồ án'
sheet['A66'] = (
    "Tuần 15: Hoàn thiện mã nguồn, đóng gói công cụ hỗ trợ và viết hướng dẫn sử dụng.\n"
    "Tuần 16-17: Hoàn thiện báo cáo bản cứng, chuẩn bị tóm tắt, thiết kế Slide và kịch bản bảo vệ đồ án tốt nghiệp."
)

# --- 4. Final Save ---
final_filename = 'Đỗ Trung Kiên_20224869_Final.xlsx'
wb.save(final_filename)
print(f"Successfully saved final plan as {final_filename}")
