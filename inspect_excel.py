import openpyxl
import io
import sys

# Set encoding to utf-8 for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

wb = openpyxl.load_workbook('Nguyễn Trung Long_20224874.xlsx')
sheet = wb.active

print(f"Sheet Title: {sheet.title}")
count = 0
for row in sheet.iter_rows():
    for cell in row:
        if cell.value is not None:
            print(f"Cell {cell.coordinate}: {cell.value}")
            count += 1
            if count > 30:
                break
    if count > 30:
        break
