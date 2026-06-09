# D11 表格解析与降级策略

## 目标

D11 的目标是先建立表格的统一表示，而不是立刻深做某个 PDF 表格识别引擎。

企业级 RAG 中，表格不能简单当作普通文本。财报、合同、产品清单、技术参数和统计报表
大量依赖表格。如果表格结构被破坏，模型很容易读错行列关系。

## 当前设计

新增 `TableBlock`：

- `rows`：二维表格数据
- `caption`：表格标题或说明
- `page_number`：来源页码
- `section_path`：所属章节路径
- `metadata`：额外解析信息

同时提供转换函数：

- `table_to_markdown`
- `table_to_csv`
- `table_to_json`
- `table_to_chunk`

## 为什么 Markdown 是默认降级格式

Markdown 表格有三个优点：

1. 人类可读，适合 README、报告和 bad case 分析。
2. LLM 容易理解，适合进入上下文。
3. 可以保留基本行列结构，优于把表格拍平成普通文本。

当结构化表格不可用时，至少要保留 caption、页码、章节路径和一个可读 Markdown 表达。

## 后续计划

- Docling/MinerU 输出表格时，统一转成 `TableBlock`。
- D13 Golden Dataset 中加入表格样例。
- D14 解析对比报告中统计表格保留情况。
