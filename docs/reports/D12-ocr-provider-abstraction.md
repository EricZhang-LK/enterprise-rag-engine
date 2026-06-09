# D12 OCR Provider 抽象

## 目标

D12 的目标是给扫描版 PDF 和图片类文档预留 OCR 能力边界。

基础 PDF 文本解析器依赖 PDF 内部已有文本层。扫描版 PDF 通常只有图片，没有可抽取文本，
所以 `pypdf` 无法直接得到内容。这类文档需要 OCR。

## 为什么先做 Provider 抽象

OCR 方案很多，例如：

- Tesseract
- PaddleOCR
- 云厂商 OCR
- Docling/MinerU 内置 OCR 能力

如果在 parser 里直接写死某一个 OCR 实现，后续替换成本会很高。更好的方式是先定义：

```text
BaseOCRProvider -> OCRResult -> OcrDocumentParser -> ParseResult
```

这样不同 OCR 后端只要实现 `extract_text`，就能进入统一文档管道。

## 当前实现

新增：

- `OCRResult`
- `OcrStatus`
- `BaseOCRProvider`
- `OcrDocumentParser`

`OcrDocumentParser` 负责把 OCR provider 返回的结果转换为：

- `Document`
- `DocumentChunk`
- `ChunkMetadata`
- `ParseResult`

## 当前边界

当前不接真实 OCR 引擎，只通过 fake provider 测试以下场景：

- 部分页成功，部分页失败
- provider 返回空结果
- provider 自身异常

真实 OCR 集成留到后续扫描件样例和复杂文档阶段。
