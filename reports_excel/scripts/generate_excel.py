import pandas as pd
from datetime import datetime

data = [
    {"STT": 1, "Giai đoạn": "Tìm hiểu và thiết lập môi trường", "Chi tiết công việc": "Nghiên cứu các bài báo liên quan, thiết lập môi trường (giải quyết xung đột thư viện NumPy, PyTorch), tinh chỉnh code để chạy tương thích.", "Trạng thái": "Hoàn thành", "Tiến độ (%)": 100},
    {"STT": 2, "Giai đoạn": "Triển khai mã nguồn mô hình", "Chi tiết công việc": "Cài đặt và chạy thử nghiệm các mô hình Image Fusion (SPDFusion, TUFusion, WaveFusion, VDMUFusion, CDDFuse, MMIF-DDFM, SFMFusion, BSAFusion...).", "Trạng thái": "Hoàn thành", "Tiến độ (%)": 100},
    {"STT": 3, "Giai đoạn": "Tối ưu hóa thuật toán", "Chi tiết công việc": "Áp dụng phương pháp lấy mẫu DDIM cho các mô hình Diffusion, giảm thời gian suy luận (inference time), tối ưu hóa luồng chạy song song.", "Trạng thái": "Hoàn thành", "Tiến độ (%)": 100},
    {"STT": 4, "Giai đoạn": "Đánh giá chất lượng ảnh (Metrics)", "Chi tiết công việc": "Cài đặt hơn 20 độ đo (PSNR, SSIM, FMI...) và thực hiện đánh giá đồng loạt trên các tập dữ liệu ảnh y tế (PET, SPECT, CT).", "Trạng thái": "Hoàn thành", "Tiến độ (%)": 100},
    {"STT": 5, "Giai đoạn": "Viết báo cáo đánh giá và cơ sở lý thuyết", "Chi tiết công việc": "Tổng hợp kết quả đánh giá, viết báo cáo LaTeX trình bày công thức toán học các độ đo và phân tích kiến trúc của các mô hình.", "Trạng thái": "Đang thực hiện", "Tiến độ (%)": 80},
    {"STT": 6, "Giai đoạn": "Viết và hoàn thiện quyển đồ án", "Chi tiết công việc": "Tổng hợp các chương lý thuyết, thực nghiệm và kết luận. Chuẩn bị định dạng chuẩn để nộp và bảo vệ.", "Trạng thái": "Đang thực hiện", "Tiến độ (%)": 20},
]

df = pd.DataFrame(data)

# Create a Pandas Excel writer using openpyxl as the engine
writer = pd.ExcelWriter("Bao_cao_tien_do_Do_Trung_Kien.xlsx", engine="openpyxl")
df.to_excel(writer, sheet_name="Tiến độ", index=False)

# Adjust column widths for better view
worksheet = writer.sheets["Tiến độ"]
for column_cells in worksheet.columns:
    length = max(len(str(cell.value)) for cell in column_cells)
    worksheet.column_dimensions[column_cells[0].column_letter].width = length + 2

writer.close()
print("Đã tạo file Bao_cao_tien_do_Do_Trung_Kien.xlsx thành công!")
