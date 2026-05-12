# LaTeX Report — ĐATN CDDFuse Variants

## Cấu trúc

```
report_latex/
├── main.tex              ← entry point, \include các chapter
├── bibliography.bib      ← danh sách trích dẫn (BibTeX)
├── chapters/
│   ├── 00_titlepage.tex
│   ├── 01_abstract.tex
│   ├── 02_intro.tex       ← Chương 1: Mở đầu
│   ├── 03_background.tex  ← Chương 2: Cơ sở lý thuyết
│   ├── 04_method.tex      ← Chương 3: Phương pháp đề xuất
│   ├── 05_results.tex     ← Chương 4: Thí nghiệm & kết quả
│   ├── 06_discussion.tex  ← Chương 5: Thảo luận
│   ├── 07_conclusion.tex  ← Chương 6: Kết luận
│   └── A_appendix.tex     ← Phụ lục
├── figures/              ← PNG/PDF hình ảnh (\graphicspath)
└── tables/               ← .tex bảng riêng (optional)
```

## Build

### Yêu cầu
- TeX Live hoặc MiKTeX (đầy đủ)
- Font Vietnamese (`vietnamese` babel)

### Compile

```bash
cd report_latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex          # 2 lần để cross-ref đúng
```

Output: `main.pdf`.

### Hoặc dùng `latexmk` (auto)
```bash
latexmk -pdf main.tex
```

### VSCode
Cài extension **LaTeX Workshop**, sẽ tự build khi save.

## Các marker `% TODO:`

File chapter có sẵn skeleton + `% TODO:` ở các chỗ cần điền nội dung cụ thể. Bám theo:
- `results_v2/PROGRESS.md` — single source of truth về thí nghiệm
- `results_v2/_stats/*/REPORT.md` — chi tiết stats từng variant
- `results_v2/_stats/*/significance_*.csv` — bảng raw

## Tips

- **Bảng số liệu**: copy từ `_stats/*/REPORT.md` (đã có format markdown), convert thành `booktabs` LaTeX
- **Hình loss curve**: dùng `results_v2/CDDFuse-Combined-Paper-MIF/train_history.json` + matplotlib → save PNG vào `figures/`
- **Citation paper gốc**: đã có `zhao2023cddfuse` trong `bibliography.bib`, dùng `\cite{zhao2023cddfuse}`
- **Cross-ref**: dùng `\ref{tab:moduleA}`, `\ref{ch:results}` thay vì gõ số trực tiếp

## Sửa thông tin cá nhân

File `chapters/00_titlepage.tex` — điền lớp + GVHD.
