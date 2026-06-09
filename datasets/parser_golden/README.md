# Parser Golden Dataset

这个目录用于沉淀文档解析评测样例。

当前阶段先建立 JSONL schema 和样例索引，不直接提交大体积 PDF/Docx 文件。后续可以把小型
公开样例或自造样例放到 `samples/`，并在 `cases.jsonl` 中登记预期结果。

每一行是一个 JSON 对象，表示一个解析样例：

```json
{
  "id": "markdown-basic-headings",
  "source_uri": "datasets/parser_golden/samples/basic.md",
  "document_type": "markdown",
  "parser": "MarkdownParser",
  "expectation": {
    "min_chunk_count": 3,
    "required_text": ["Product Guide"],
    "required_section_paths": [["Product Guide", "Install"]]
  },
  "notes": "Markdown heading hierarchy should be preserved."
}
```

## 当前目标

- 普通 PDF
- 表格 PDF
- Word
- Markdown
- 扫描件样例
- 结构化 PDF 样例

## 后续指标

- 解析成功率
- 平均耗时
- chunk 数量
- 页码保留率
- section path 保留率
- 表格结构保留情况
