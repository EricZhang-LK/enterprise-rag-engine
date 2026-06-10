# D15 Recursive Splitter 学习笔记

## 今日目标

D15 的目标是实现 `RecursiveSplitter`，让已经解析完成的 `Document` 可以被切成适合检索、引用和后续评测的 `DocumentChunk`。

今天先使用字符长度作为预算单位。这样做是为了先建立稳定的分块算法骨架，D16 再把字符预算升级为 token 预算。

## 为什么 RAG 分块不是切字符串

RAG 中的 chunk 是检索系统的最小召回单元。一个 chunk 的质量，会直接影响后续三个环节：

- 检索：chunk 太短会丢上下文，chunk 太长会稀释 query 与文本之间的相似度。
- 重排：reranker 只能在候选 chunk 上工作，候选本身切坏了，重排也很难救回来。
- 引用：如果 chunk 没有 `source_uri`、`document_id`、`start_char`、`end_char` 等元数据，答案就很难追溯。

所以分块不是简单的 `text[:800]`。企业级 RAG 的分块策略需要同时考虑：

- 语义完整性：尽量不要把一个段落、句子、表格或步骤切断。
- 长度可控：每个 chunk 必须适合 embedding 模型和后续上下文窗口。
- 可追溯性：chunk 必须知道自己来自哪个文档、哪个位置。
- 稳定性：同一份文档重复处理时，chunk 结果应尽量稳定，便于缓存和回归评测。

## Recursive Splitter 的核心思想

Recursive Splitter 的思想是：先尝试强语义边界，失败后再逐步使用弱边界。

当前实现使用的边界顺序是：

```text
段落 -> 换行 -> 中文句号/感叹号/问号 -> 英文句末 -> 分号 -> 逗号 -> 空格 -> 字符窗口
```

这条顺序体现了一个工程原则：优先保留人类写作结构，最后才使用机械切分兜底。

例如：

```text
第一段介绍背景。

第二段解释方案。

第三段给出结论。
```

如果长度预算允许，分块器会尽量以段落为单位；如果某个段落本身过长，再继续按句子或空格拆分；如果是一长串没有任何分隔符的文本，最后才退化为固定字符窗口。

## Overlap 的作用与风险

`overlap_chars` 的作用是让相邻 chunk 保留一小段重叠文本，降低边界处语义断裂的风险。

比如一个定义刚好跨越 chunk 边界：

```text
向量数据库的作用是...
```

如果前半句在 chunk A，后半句在 chunk B，检索时可能两个 chunk 都不够完整。overlap 可以让边界附近的信息在相邻 chunk 中重复出现。

但 overlap 也有代价：

- chunk 数量增加，embedding 成本上升。
- 重复内容变多，检索结果可能更冗余。
- 如果 overlap 太大，平均 chunk 长度和上下文预算都会失控。

所以 overlap 不是越大越好。后续 D20 的 chunk 评测会统计平均长度、P95 长度、重叠率和 bad case。

## 今日实现边界

今天的 `RecursiveSplitter` 已完成：

- 实现 `BaseSplitter` 抽象基类。
- 支持 `max_chars` 和 `overlap_chars`。
- 支持递归边界切分。
- 支持长文本字符窗口兜底。
- 生成 `DocumentChunk`。
- 保留 `source_uri`、`document_id`、`content_hash`、`start_char`、`end_char`。

暂未处理：

- token 计数，D16 实现。
- Parent-Child Chunk，D17 实现。
- 语义相似度分块，D18 实验。
- 分块质量指标脚本，D20 实现。

## 面试表达

可以这样描述今天的设计：

> 我没有直接按固定长度切字符串，而是实现了一个递归分块器。它会优先按段落、换行、句子、标点和空格等边界切分，最后才退化为固定窗口。每个 chunk 都保留原文 offset 和 content hash，为后续 citation、缓存、评测和 bad case 回归打基础。

