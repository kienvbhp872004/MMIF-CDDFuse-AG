import openpyxl
import io
import sys

# Set encoding to utf-8 for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

wb = openpyxl.load_workbook('Đỗ Trung Kiên_20224869_Final.xlsx')
sheet = wb.active

for row in range(20, 38):
    val = sheet[f'A{row}'].value
    if val:
        print(f"Cell A{row}: {val}")
