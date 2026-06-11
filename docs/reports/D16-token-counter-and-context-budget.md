# D16 Token 计数与上下文预算学习笔记

## 今日目标

D16 的目标是建立 `TokenCounter` 和 `TokenBudget`，让分块系统从“字符长度可控”逐步走向“模型上下文预算可控”。

D15 的 `RecursiveSplitter` 使用 `max_chars`，这适合先搭建分块算法骨架。但真实 LLM 和 embedding 模型限制的是 token，不是字符。

## 字符长度为什么不够

字符数和 token 数不是一回事。

英文中，一个词可能是一个 token，也可能被拆成多个 token：

```text
retrieval-augmented generation
```

中文中，一个汉字、词语、标点在不同 tokenizer 中也可能有不同切分结果。

所以如果只用字符数控制 chunk，会出现两个问题：

- 看起来短的文本，实际 token 可能超预算。
- 看起来长的文本，实际 token 可能还能放下，导致上下文窗口浪费。

在企业级 RAG 中，token 预算会影响：

- embedding 请求是否超长。
- rerank 输入是否超长。
- prompt context 是否被截断。
- 模型输出是否有足够空间。
- 成本估算是否准确。

## TokenCounter 的设计取舍

第一版实现的是一个无外部依赖的稳定估算器，而不是模型精确 tokenizer。

它的目的不是替代 `tiktoken`、Qwen tokenizer 或 BGE tokenizer，而是先给项目建立一个统一边界：

```text
text -> TokenCounter -> token_count
```

后续如果接入模型精确 tokenizer，只需要替换这个边界里的实现，而不是修改 splitter、retriever、context builder 和 eval 脚本。

当前估算规则：

- 中文字符按单个 token-like span 计数。
- 英文单词和数字按词计数。
- 标点符号按单独 token-like span 计数。
- 保留每个 token-like span 的 `start` 和 `end` offset，方便截断和调试。

## 成熟 tokenizer 选型

企业级项目不能只依赖估算器。修订后，项目新增了两个成熟 tokenizer 适配器：

| 适配器 | 适用模型 | 优点 | 代价 | 项目定位 |
|---|---|---|---|---|
| `TiktokenTokenCounter` | OpenAI / OpenAI-compatible 模型 | 官方、速度快、适合生产计费和上下文预算 | 主要覆盖 OpenAI tokenizer 体系 | OpenAI 路线首选 |
| `HuggingFaceTokenCounter` | Qwen、BGE、bge-reranker、其他开源模型 | 覆盖面广，能跟具体模型 tokenizer 对齐 | 依赖较重，首次加载可能下载模型 tokenizer 文件 | 国产/开源模型路线首选 |
| `TokenCounter` | 本地测试、CI、无网络环境 | 无依赖、稳定、快 | 非模型精确 | fallback，不作为生产最终口径 |

最终选择：

- 如果后续使用 OpenAI embedding 或 OpenAI-compatible chat model，默认使用 `TiktokenTokenCounter`。
- 如果后续使用 Qwen、BGE-M3 或 bge-reranker，默认使用 `HuggingFaceTokenCounter`。
- `TokenCounter` 只用于本地 fallback、单元测试和没有安装 optional dependency 的环境。

依赖被放入 optional extra：

```powershell
python -m pip install -e ".[dev,tokenization]"
```

这样基础项目仍然轻量，生产或实验环境可以显式安装成熟 tokenizer。

## TokenBudget 的作用

上下文窗口不是全部留给 chunk 的。一个 RAG 请求通常至少包含：

- system prompt
- instruction prompt
- user query
- retrieved chunks
- expected output

例如一个 8192 token 的模型窗口，如果预留：

```text
output: 1024
system: 512
prompt: 256
```

那么真正留给用户问题和检索 chunk 的预算是：

```text
8192 - 1024 - 512 - 256 = 6400
```

这就是 `TokenBudget.available_input_tokens` 的意义。

如果计划召回 8 个 chunk，可以粗略分配：

```text
6400 / 8 = 800 tokens per chunk
```

后续 D17 的 Parent-Child Chunk 和 D45 的 Token Budget Manager 都会用到这个思想。

## 今日实现边界

今天完成：

- `TokenCounter.count`
- `TokenCounter.count_many`
- `TokenCounter.spans`
- `TokenCounter.fits`
- `TokenCounter.truncate`
- `TiktokenTokenCounter`
- `HuggingFaceTokenCounter`
- `TokenBudget.available_input_tokens`
- `TokenBudget.allocate_per_chunk`

暂未完成：

- 将 `RecursiveSplitter` 从 `max_chars` 改造成 `max_tokens`。
- 按模型类型区分 embedding token 预算和 generation token 预算。
- 上下文拼接阶段的动态截断。

这些内容会在 W3 后续和 W7 上下文工程中继续演进。

## 面试表达

可以这样描述今天的设计：

> 我在分块模块中没有把字符数当成最终标准，而是抽出了 TokenCounter 和 TokenBudget。当前实现是无依赖的稳定估算器，用于测试和本地开发；后续可以替换为模型精确 tokenizer。TokenBudget 会显式预留输出、system prompt 和 instruction prompt 的空间，避免检索 chunk 把整个上下文窗口占满。
