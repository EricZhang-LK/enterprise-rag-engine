# D13 文档解析 Golden Dataset

## 目标

D13 的目标是让文档解析质量开始可评估。

到目前为止，我们已经有了 PDF、Markdown、Docx、结构化 PDF、表格和 OCR 的基础能力。
但如果没有 Golden Dataset，我们只能说“看起来能解析”。这不够。

Golden Dataset 的价值是把预期结果写下来，让每次修改 parser 后都能检查是否退化。

## 当前实现

新增：

- `ParserGoldenCase`
- `ParserGoldenExpectation`
- `ParserGoldenDataset`
- `load_parser_golden_dataset`
- `summarize_parser_golden_dataset`

样例文件：

```text
datasets/parser_golden/cases.jsonl
```

当前包含 5 类样例索引：

- 普通 PDF
- Markdown 标题层级
- Word Heading 层级
- 表格 PDF
- 扫描版 PDF OCR

## 当前边界

今天只提交 JSONL 样例索引和 schema，不提交真实大文件。真实样例会在后续逐步加入
`datasets/parser_golden/samples/`。

## 后续指标

- 解析成功率
- 平均耗时
- 页码保留率
- section path 保留率
- 表格保留情况
- OCR 成功率

这些指标会在 D14 的解析对比报告中开始出现。
