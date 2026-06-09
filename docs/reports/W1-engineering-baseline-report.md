# W1 工程基线报告

## 范围

W1 的目标是为 `enterprise-rag-engine` 建立工程基础。

这一周刻意不急着写 RAG 业务逻辑，而是先建立稳定的项目边界。后续的文档解析、分块、
检索、评测和 API 工作流，都要建立在这些基础设施之上。

## 已交付内容

| 领域 | 交付物 |
|---|---|
| 项目结构 | `src layout`、tests、docs、scripts、datasets |
| 工具链 | `ruff`、`mypy strict`、`pytest`、`pytest-cov`、`pre-commit` |
| 数据契约 | 文档、chunk、解析结果、检索结果等 Pydantic 模型 |
| 行为契约 | 基于 ABC 的 parser、splitter、embedder、vector store、retriever、evaluator 接口 |
| 服务骨架 | FastAPI app factory 和 `/health` 接口 |
| 运行基础 | settings、`.env.example`、日志配置、统一业务异常 |
| 架构文档 | ADR-001 核心契约决策 |

## 质量门禁

最近一次验证结果：

```text
ruff: passed
mypy: passed
pytest: 12 passed
coverage: >95%
```

当前 Windows 本地环境需要把工具缓存放到项目目录外：

```powershell
$env:RUFF_CACHE_DIR="D:\code\codex_learn\.tool-cache\ruff"
$env:COVERAGE_FILE="D:\code\codex_learn\.tool-cache\coverage\.coverage.enterprise-rag-engine"
mypy --cache-dir "D:\code\codex_learn\.tool-cache\mypy" src tests
```

## 设计说明

项目使用 Pydantic 模型作为数据契约，因为 RAG 系统需要在解析、分块、检索、API 响应
和评测之间传递明确、可校验、可序列化的对象。

项目使用抽象基类作为组件契约，因为这个项目会逐步演进成偏框架化的开源项目。显式继承
能让实现关系更清楚，也方便后续在基类中沉淀通用逻辑，例如计时、日志、校验和错误包装。

## 下一步

W2 将开始文档管道建设，第一步是 PDF 文本解析。Parser 实现必须返回 `ParseResult`，
并从一开始就保留来源、页码和基础 metadata。
