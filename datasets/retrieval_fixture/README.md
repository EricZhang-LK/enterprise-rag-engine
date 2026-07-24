# 检索评测固定排名样例

该目录包含 10 条离线固定排名样例，用于验证 `retrieval_eval.py` 的数据读取、指标计算和报告表格。

每行 JSONL 代表一个 query，包含人工标注的 `relevant_chunk_ids`，以及 `dense`、`bm25`、`hybrid_rrf`、`hybrid_rerank` 四组预先记录的 chunk 排名。

这是**评测夹具**，不是由真实 BGE-M3、Qdrant 或 Cross Encoder 在生产语料上运行得到的 benchmark。它不能用于声称模型效果、延迟或线上收益；真实 Golden QA 会在 W8 扩展到至少 50 条，并接入真实检索链路。
