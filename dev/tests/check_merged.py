import openpyxl

wb = openpyxl.load_workbook('Đỗ Trung Kiên_20224869.xlsx')
sheet = wb.active

print("Merged cells:")
for range_ in sheet.merged_cells.ranges:
    print(f"Range: {range_}")
