# edurag-qa-system

学科在线答疑系统的最小可运行复现：**FAQ 阈值直答 + 知识库双路召回（向量 / BM25）融合排序**，
附命中率统计、反馈沉淀与 Docker 部署。

技术栈：Python / FastAPI / scikit-learn(TF-IDF 向量) / 自实现 BM25 / JSONL 日志

## 对应原始项目的设计

| 原项目（EduRAG） | 本复现 |
| --- | --- |
| Milvus 向量检索 | TF-IDF 向量检索（接口一致，可换成 Milvus / Chroma） |
| BM25 关键词检索 | 同目录下自实现的 BM25Okapi（约 40 行） |
| MySQL 存 FAQ / Redis 缓存 | JSON 文件 + 进程内缓存（接口留好了，可替换） |
| 500+ 条高频问题 FAQ 库 | `data/faq.json` 内置 20 条示例，可继续扩充 |
| 相似度阈值 ≥0.85 直出标准答案 | `FAQ_THRESHOLD` 默认 0.85，可配置 |

## 功能

- `POST /ask`：提问 → FAQ 命中（阈值以上）直接返回标准答案；未命中走知识库双路召回
- `GET /faqs`：查看 FAQ 库
- `POST /feedback`：对答案点「有用 / 没用」，沉淀为待补充问题
- `GET /stats`：FAQ 命中率、平均响应、Top 未命中问题（对应原项目的准确率量化）
- `GET /healthz`

## 快速开始

```bash
pip install -r requirements.txt
uvicorn app.api:app --port 8001
# http://localhost:8001/
```

Docker：`docker compose up -d --build`（端口 8001）

## 测试

```bash
python -m pytest -q
```
