import openpyxl

wb = openpyxl.load_workbook('Đỗ Trung Kiên_20224869_Final.xlsx')
sheet = wb.active

# --- Update Sections 3.1 - 3.4 (Using top-left cells of merged ranges) ---

# 3.1 Kiến thức (Range A25:K27)
sheet['A25'] = (
    "- Nâng cao kiến thức chuyên sâu về xử lý ảnh y tế đa phương thức (Medical Imaging: MRI, CT, PET, SPECT).\n"
    "- Làm chủ các kỹ thuật phân tách đa thang đo (Multiscale Decomposition) như DWT, NSST và các phép lọc bảo toàn biên (Edge-preserving Filters).\n"
    "- Hiểu rõ nền tảng toán học về Fusion Rules (Luminance balance, Contrast preservation) và lý thuyết bảo toàn thông tin (Information Theory).\n"
    "- Kiến thức về hệ thống các chỉ số đánh giá khách quan (Qab/f, FMI, SSIM, VIF) để định lượng độ chi tiết và độ trung thực của ảnh trộn."
)

# 3.2 Công nghệ (Range A29:K31)
sheet['A29'] = (
    "- Ngôn ngữ lập trình chính: Python.\n"
    "- Thư viện xử lý ảnh: OpenCV, Scikit-image, PyWavelets, NSST (Non-subsampled Shearlet Transform toolbox).\n"
    "- Triển khai mô hình SOTA: Các giải pháp trộn ảnh kết hợp giữa kiến trúc phân tách cổ điển và các thành phần học máy nền tảng (DenseFuse/Deep-Fuse concept).\n"
    "- Công cụ đánh giá Metrics:\n"
    "   + Metrics cấu trúc: SSIM, MS-SSIM.\n"
    "   + Metrics thông tin: Information Entropy, Mutual Information.\n"
    "   + Metrics tần số: Average Gradient, Spatial Frequency.\n"
    "- Quản lý và báo cáo: Git, LaTeX, Overleaf."
)

# 3.3 Kỹ năng (Range A33:K33)
sheet['A33'] = (
    "- Kỹ năng tư duy giải thuật: Khả năng toán học hóa các bài toán về tích hợp thông tin đa nguồn.\n"
    "- Kỹ năng triển khai nghiên cứu (Research Execution): Khả năng đọc hiểu bài báo khoa học và cài đặt lại các kiến trúc SOTA tiên tiến.\n"
    "- Kỹ năng phân tích phản biện: Đánh giá mối quan hệ giữa các độ đo kỹ thuật và khả năng quan sát thực tiễn của bác sĩ chuyên khoa.\n"
    "- Kỹ năng quản lý dữ liệu lớn: Xử lý và chuẩn hóa các bộ dữ liệu y tế khổng lồ (Harvard Atlas Dataset)."
)

# 3.4 Sản phẩm (Range A35:K37)
sheet['A35'] = (
    "- Tài liệu: Báo cáo Đồ án tốt nghiệp LaTeX, Slide thuyết trình chuyên nghiệp, Video demo quy trình xử lý ảnh.\n"
    "- Phần mềm / Mã nguồn:\n"
    "   + Framework hoàn chỉnh tích hợp 5+ thuật toán fussion mạnh mẽ (DWT, NSST, GFF, LP, Cải tiến Hybrid).\n"
    "   + Bộ công cụ Metrics Suite tự động kết xuất báo cáo so sánh dưới dạng CSV/Excel.\n"
    "- Dữ liệu: Bộ Dataset được chuẩn hóa và phân loại theo các cặp bệnh lý (Brain Tumors, Alzheimer, Stroke).\n"
    "- Sản phẩm khoa học: Đề xuất một cấu hình luật trộn cải tiến có hiệu suất vượt trội so với các phương pháp cơ sở."
)

# --- Update Implementation Plan (Weekly - Accelerated) ---

# Nội dung 1 (1-4)
sheet['A44'] = 'Nội dung 1: Khảo sát tài liệu và triển khai thử nghiệm các mô hình SOTA'
sheet['A46'] = (
    "Tuần 1-2: Nghiên cứu ứng dụng Image Fusion trong y khoa. Thu thập và tiền xử lý bộ dữ liệu Harvard Atlas.\n"
    "Tuần 3-4 (Hiện tại): Triển khai cài đặt các thuật toán SOTA cơ sở: DWT, NSST, Laplacian Pyramid.\n"
    "Nghiên cứu cấu trúc mã nguồn và đánh giá sơ bộ hiệu năng thực tế của các mô hình này."
)

# Nội dung 2 (5-7)
sheet['A49'] = 'Nội dung 2: Xây dựng hệ thống đánh giá toàn diện và đề xuất cải tiến'
sheet['A51'] = (
    "Tuần 5-7: Hoàn thiện bộ Metrics Suite (Entropy, VIF, SSIM, Qab/f). \n"
    "Phân tích nhược điểm của các mô hình SOTA cơ sở để đưa ra hướng cải tiến luật trộn trọng số (Weighting Optimization)."
)

# Nội dung 3 (8-10)
sheet['A54'] = 'Nội dung 3: Phát triển thuật toán cải tiến Hybrid Fusion'
sheet['A56'] = (
    "Tuần 8-10: Triển khai luật trộn cải tiến dựa trên Adaptive Weighting hoặc Guided Filter Enhancement.\n"
    "Thực hiện đo lường tỉ mỉ đối với ảnh y tế đa kênh để đảm bảo độ bảo toàn biên đạt tối ưu."
)

# Nội dung 4 (11-13)
sheet['A59'] = 'Nội dung 4: Thử nghiệm thực tế và Đánh giá thống kê diện rộng'
sheet['A61'] = (
    "Tuần 11-13: Chạy thực nghiệm trên 100+ mẫu bệnh lý khác nhau. \n"
    "Kết xuất số liệu so sánh chi tiết giữa phương pháp cải tiến và các phương pháp SOTA để kiểm chứng đột phá."
)

# Nội dung 5 (14-17)
sheet['A64'] = 'Nội dung 5: Viết báo cáo học thuật và Đóng gói sản phẩm'
sheet['A66'] = (
    "Tuần 14-17: Đóng gói Framework, viết hướng dẫn vận hành.\n"
    "Hoàn thiện báo cáo đồ án tốt nghiệp bằng LaTeX chuẩn khoa học và chuẩn bị slide bảo vệ."
)

# --- Final Save ---
final_filename = 'Đỗ Trung Kiên_20224869_v3.xlsx'
wb.save(final_filename)
print(f"Successfully saved revised plan as {final_filename}")
