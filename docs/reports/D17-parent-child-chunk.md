# D17 Parent-Child Chunk 学习笔记

## 今日目标

D17 的目标是实现 `ParentChildSplitter`，解决 RAG 分块中的经典矛盾：

- 小 chunk 更适合向量检索，因为语义更聚焦。
- 大 chunk 更适合回答生成，因为上下文更完整。

Parent-Child Chunk 的核心思想是：检索时用 child chunk，生成时回到 parent chunk。

## 为什么需要 Parent-Child Chunk

如果只使用大 chunk：

- 优点：上下文完整，模型更容易读懂来龙去脉。
- 缺点：embedding 表示会被大量无关信息稀释，召回精度下降。

如果只使用小 chunk：

- 优点：语义聚焦，向量检索更容易命中。
- 缺点：回答生成时上下文太碎，容易丢定义、前提、例外条件和跨句关系。

Parent-Child Chunk 让这两个目标拆开：

```text
Document
  -> Parent Chunk: 较大，上下文完整，用于生成
      -> Child Chunk: 较小，语义聚焦，用于检索
```

## 检索链路中的工作方式

典型流程是：

```text
1. 文档被切成 parent chunks
2. 每个 parent 再切成 child chunks
3. 向量库只索引 child chunks
4. 查询命中 child chunks
5. 根据 child.parent_id 找回 parent chunk
6. 将 parent chunk 放入最终上下文
```

这样做的收益是：

- embedding 召回阶段看的是小块，匹配更准。
- LLM 生成阶段看的是大块，信息更完整。
- bad case 分析更清楚：可以判断问题出在 child 召回，还是 parent 上下文不足。

## 今日实现

`ParentChildSplitter` 内部复用了 D15 的 `RecursiveSplitter`：

- `parent_splitter` 生成较大的 parent chunks。
- `child_splitter` 在每个 parent 内继续生成较小的 child chunks。
- child chunk 的 `parent_id` 指向 parent chunk 的 `id`。
- child chunk 的 `start_char` 和 `end_char` 会从 parent 内相对位置映射回原始文档位置。

这点很重要。否则后续 citation 和 debug 时，只知道 child 在 parent 内的位置，不知道它在原文中的位置。

## 当前边界

今天完成：

- `ParentChildSplitter`
- parent/child 两级切分
- `parent_id` 关联
- child offset 映射回原始文档
- 4 个单元测试

暂未完成：

- 向量库只索引 child、文档库保存 parent 的存储策略。
- parent 去重与多 child 命中合并。
- parent 过大时的上下文预算裁剪。
- 与 tokenizer 精确预算结合。

这些会在 W5-W7 的检索、rerank 和 context builder 中继续演进。

## 面试表达

可以这样描述今天的设计：

> 我实现了 Parent-Child 分块。child chunk 用于向量检索，保持语义聚焦；parent chunk 用于最终上下文，保持信息完整。child 通过 parent_id 指向 parent，并且保留原文 start/end offset，这样后续可以做 citation、debug 和 bad case 回归。

