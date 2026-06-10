# W2 文档解析对比报告与 Bad Case 记录

## 阶段目标

W2 的目标是建立企业级 RAG 的文档解析入口。

这一阶段不追求一次性解决所有复杂文档，而是先让不同类型文档统一进入同一套契约：

```text
Parser -> ParseResult -> Document + DocumentChunk + ChunkMetadata
```

这样后续分块、检索、引用和评测都可以建立在稳定的解析结果上。

## 解析能力对比

| 模块 | 适用场景 | 优点 | 当前限制 |
|---|---|---|---|
| `PdfTextParser` | 普通文本 PDF | 简单、轻量、保留页码 | 不擅长复杂版面、表格和扫描件 |
| `MarkdownParser` | Markdown 文档 | 标题层级清晰，适合保留 section path | 依赖文档本身规范 |
| `DocxParser` | Word 文档 | 可利用 Heading 样式恢复章节路径 | 对复杂表格、页眉页脚尚未深做 |
| `StructuredPdfParser` | 复杂 PDF | 通过 Docling 导出结构化 Markdown | Docling 为 optional dependency，真实样例后续补充 |
| `TableBlock` | 表格降级 | 支持 Markdown/CSV/JSON/table chunk | 尚未接真实表格识别器 |
| `OcrDocumentParser` | 扫描版 PDF / 图片文档 | OCR 后端可替换 | 当前只定义 provider 边界，未接真实 OCR |

## 当前质量基线

Parser Golden Dataset 当前包含 5 类样例索引：

- 普通 PDF
- Markdown 标题层级
- Word Heading 层级
- 表格 PDF
- 扫描版 PDF OCR

当前统计：

```text
case_count: 5
table_cases: 1
type:docx: 1
type:markdown: 1
type:pdf: 3
```

## Bad Case 模板

后续每遇到一个解析失败或效果不佳的文档，按下面格式记录：

```text
Case ID:
Source:
Document Type:
Parser:
Expected:
Actual:
Impact:
Root Cause:
Fix Plan:
Regression Test:
```

## 已知 Bad Cases

### BC-001: 扫描版 PDF 无文本层

- Source: scanned PDF
- Parser: `PdfTextParser`
- Expected: 提取正文文本
- Actual: `pypdf` 无法抽取文本
- Impact: 无法进入后续分块和检索
- Root Cause: 文档只有图片，没有文本层
- Fix Plan: 使用 `OcrDocumentParser` 接入 OCR provider
- Regression Test: `scanned-pdf-ocr` golden case

### BC-002: 复杂 PDF 表格结构丢失

- Source: table PDF
- Parser: `PdfTextParser`
- Expected: 保留行列关系
- Actual: 表格可能被抽成普通文本
- Impact: 模型可能读错指标和值的对应关系
- Root Cause: 基础文本抽取不理解表格结构
- Fix Plan: 使用 `StructuredPdfParser` 或后续表格识别器输出 `TableBlock`
- Regression Test: `pdf-table-fallback` golden case

### BC-003: Word 文档未使用 Heading 样式

- Source: manually formatted Word document
- Parser: `DocxParser`
- Expected: 恢复章节路径
- Actual: 如果作者只用加粗/字号模拟标题，当前 parser 无法识别
- Impact: section path 缺失，影响后续分块上下文
- Root Cause: 当前只读取 Word style name
- Fix Plan: 后续可增加字体大小、加粗、编号规则等启发式识别
- Regression Test: 后续加入 malformed-docx-heading golden case

## 下一阶段衔接

W3 将进入 chunk 分块系统。W2 的核心产物会成为 W3 的输入：

- `Document.content`
- `DocumentChunk`
- `ChunkMetadata.page_number`
- `ChunkMetadata.section_path`
- `TableBlock`
- Parser Golden Dataset

分块系统必须在不破坏这些 metadata 的前提下工作。
