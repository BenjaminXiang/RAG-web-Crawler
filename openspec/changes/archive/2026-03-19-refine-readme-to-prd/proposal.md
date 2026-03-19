## Why

当前项目仅有一个空标题 README，缺乏任何功能定义、架构描述和实现路线。需要将 "RAG-web-Crawler" 的模糊概念细化为一个结构清晰、可执行的产品需求文档（PRD），为后续开发提供明确的功能边界、技术选型和验收标准。

## What Changes

- 定义系统核心功能：Web 爬虫引擎、文档处理管道、向量化存储、语义检索问答接口
- 明确系统边界：爬取范围控制、支持的内容格式、并发与限流策略
- 确立技术栈选型约束：Python 生态、向量数据库、LLM API 集成
- 输出完整 PRD，替代当前空 README 作为项目北极星文档

## Capabilities

### New Capabilities
- `web-crawler`: URL 列表爬取引擎，按给定 URL 列表逐一爬取页面内容（不递归跟踪链接），保留页面内超链接信息，支持速率限制和 robots.txt 遵守
- `document-processor`: 文档清洗与分块管道，将原始 HTML 转换为结构化文本块（chunks），支持多种分块策略
- `vector-store`: 向量化存储层，将文本块通过 embedding 模型转换为向量并持久化存储，支持相似度检索
- `rag-query`: RAG 检索问答接口，接受自然语言查询，检索相关文档块，调用 LLM 生成带引用的回答

### Modified Capabilities
（无已有 capability，此为全新项目）

## Impact

- **代码**：全新项目，无现有代码受影响
- **依赖**：需引入爬虫库（如 crawl4ai/scrapy）、embedding 模型（如 OpenAI/sentence-transformers）、向量数据库（如 ChromaDB/Qdrant）、LLM API
- **API**：将暴露 CLI 和/或 REST API 用于触发爬取和执行查询
- **系统**：需考虑爬取目标站点的负载影响，遵守 robots.txt 和速率限制
