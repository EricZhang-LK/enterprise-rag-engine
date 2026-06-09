# D10 结构化 PDF 解析选型记录

## 目标

D10 的目标不是替换基础 `PdfTextParser`，而是为复杂 PDF 增加一个结构化解析入口。

基础 `pypdf` 适合快速抽取文本和页码，但对标题层级、阅读顺序、表格结构、公式和复杂版面
支持有限。企业级 RAG 中，复杂 PDF 解析结果通常需要先转换成结构化 Markdown 或 JSON，
再进入分块和检索链路。

## 选型结论

当前阶段主攻 Docling，MinerU 暂时作为后续复杂场景备选。

## 为什么选择 Docling

Docling 官方文档强调它可以把 PDF、Office、HTML、Markdown、图片等多种文档转换为统一
的 `DoclingDocument`，并支持导出 Markdown、HTML、JSON 等格式。它还重点覆盖 PDF 版面、
阅读顺序、表格结构、代码和公式等能力。

这与当前项目的目标匹配：我们需要一个结构化 PDF 解析器，把复杂 PDF 转成 Markdown，
再统一落到 `ParseResult`、`Document` 和 `DocumentChunk`。

## 为什么暂不主攻 MinerU

MinerU 也很强，尤其面向复杂文档抽取、OCR、多模态和中文资料时值得评估。但它的能力边界
更重，工程集成和运行环境成本可能更高。当前项目还在 W2 文档管道早期，先用 Docling
建立结构化解析适配器更稳。

## 当前实现

新增 `StructuredPdfParser`：

```text
source PDF -> Docling DocumentConverter -> export_to_markdown -> ParseResult
```

当前实现采用可选依赖：

```powershell
pip install -e ".[structured]"
```

如果未安装 Docling，`StructuredPdfParser` 会返回 `FAILED` 状态，并提示安装 optional
dependency。

## 后续计划

- D11：处理表格解析与降级策略。
- D13：把结构化 PDF 样例加入 Golden Dataset。
- D14：形成 Docling vs 基础 pypdf 的解析对比报告。

## 参考资料

- Docling 官方网站：https://www.docling.ai/
- Docling Features：https://docling.site/features/
- Docling Usage：https://docling-project.github.io/docling/usage/
- MinerU 文档：https://mineru.net/doc/docs/
- MinerU PyPI：https://pypi.org/project/mineru/
