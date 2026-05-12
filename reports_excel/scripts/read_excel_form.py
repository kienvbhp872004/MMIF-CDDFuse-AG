import pandas as pd
import sys

try:
    file_path = "d:\\Workspace\\Github\\Repo\\Image-Fusion\\Đỗ Trung Kiên - 20224869.xlsx"
    xl = pd.ExcelFile(file_path)
    with open("excel_structure.txt", "w", encoding="utf-8") as f:
        f.write(f"Sheet names: {xl.sheet_names}\n")
        for sheet in xl.sheet_names:
            f.write(f"--- Sheet: {sheet} ---\n")
            df = xl.parse(sheet)
            f.write(f"Columns: {df.columns.tolist()}\n")
            f.write(df.to_string() + "\n")
except Exception as e:
    with open("excel_structure.txt", "w", encoding="utf-8") as f:
        f.write(f"Error: {e}\n")
