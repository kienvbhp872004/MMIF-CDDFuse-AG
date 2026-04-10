import openpyxl
import sys

# Load the workbook
wb = openpyxl.load_workbook('Đỗ Trung Kiên_20224869.xlsx')
sheet = wb.active

# Open a file for writing in UTF-8
with open('excel_inspection_output.txt', 'w', encoding='utf-8') as f:
    for row in sheet.iter_rows(min_row=1, max_row=100):
        for cell in row:
            if cell.value is not None:
                f.write(f'{cell.coordinate}: {cell.value}\n')

print("Inspection results saved to excel_inspection_output.txt")
