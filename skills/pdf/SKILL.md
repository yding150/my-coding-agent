---
name: pdf
description: 处理 PDF 文件的专业技能，支持文本提取、创建 PDF、合并与拆分文档。当用户需要读取、生成或操作 PDF 文件时使用。
---

# PDF 处理技能规范 (PDF Processing Skill)

你现在拥有完整的 PDF 文档处理与操作能力。在执行 PDF 相关任务时，请遵循以下推荐工作流与方案：

## 1. 读取与提取 PDF 内容

**方案 1：快速纯文本提取（推荐首选）**
```bash
# 使用 pdftotext (来自 poppler-utils)
pdftotext input.pdf -          # 直接输出到标准输出
pdftotext input.pdf output.txt # 导出保存为文本文件

# 若系统中没有 pdftotext CLI，可使用 Python 脚本：
python -c "
import fitz  # PyMuPDF
doc = fitz.open('input.pdf')
for page in doc:
    print(page.get_text())
"
```

**方案 2：带元数据的逐页精细读取**
```python
import fitz  # 需要安装依赖: pip install pymupdf

doc = fitz.open("input.pdf")
print(f"总页数: {len(doc)}")
print(f"元数据: {doc.metadata}")

for i, page in enumerate(doc):
    text = page.get_text()
    print(f"--- 第 {i+1} 页 ---")
    print(text)
```

## 2. 生成与创建 PDF

**方案 1：从 Markdown 转换（最直观高效）**
```bash
# 使用 pandoc
pandoc input.md -o output.pdf

# 自定义字体与页边距
pandoc input.md -o output.pdf --pdf-engine=xelatex -V geometry:margin=1in
```

**方案 2：纯 Python 代码绘制**
```python
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

c = canvas.Canvas("output.pdf", pagesize=letter)
c.drawString(100, 750, "Hello, PDF!")
c.save()
```

**方案 3：从 HTML 网页渲染**
```bash
# 使用 wkhtmltopdf
wkhtmltopdf input.html output.pdf

# 或通过 Python pdfkit
python -c "
import pdfkit
pdfkit.from_file('input.html', 'output.pdf')
"
```

## 3. 合并 PDF 文件 (Merge)

```python
import fitz

result = fitz.open()
for pdf_path in ["file1.pdf", "file2.pdf", "file3.pdf"]:
    doc = fitz.open(pdf_path)
    result.insert_pdf(doc)
result.save("merged.pdf")
print("成功合并 PDF 文件至 merged.pdf")
```

## 4. 拆分 PDF 文件 (Split)

```python
import fitz

doc = fitz.open("input.pdf")
for i in range(len(doc)):
    single = fitz.open()
    single.insert_pdf(doc, from_page=i, to_page=i)
    single.save(f"page_{i+1}.pdf")
print(f"成功将文档拆分为 {len(doc)} 个单页文件。")
```

## 核心依赖库速查表

| 操作需求 | 推荐类库 | 安装指令 | 说明 |
|---|---|---|---|
| 读取 / 写入 / 合并 | **PyMuPDF (fitz)** | `pip install pymupdf` | 性能最高、功能最全、最常用 |
| 代码绘制版面 | **ReportLab** | `pip install reportlab` | 适合生成发票、报表凭据 |
| HTML 转 PDF | **pdfkit** | `pip install pdfkit` + wkhtmltopdf | 适合富文本排版转换 |
| 命令行快速提取 | **pdftotext** | `poppler` (通过包管理器安装) | 纯 CLI 提取，极速 |

## 最佳实践指南

1. **环境检查**：在运行相关命令或脚本前，先检查目标工具或 Python 依赖是否已安装；
2. **字符编码处理**：提取文本时务必确保 UTF-8 编码，防止中文乱码；
3. **大文件防护**：处理超多页数的大型 PDF 时，采用逐页流式迭代处理，防止内存耗尽 (OOM)；
4. **扫描件 OCR**：若常规提取返回内容为空，说明可能是扫描图片版 PDF，需告知用户或使用 `pytesseract` 进行光学字符识别。
