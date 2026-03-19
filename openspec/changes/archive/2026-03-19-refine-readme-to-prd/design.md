## Context

RAG-web-Crawler 是一个全新项目，目标是构建一条从网页爬取到 RAG 就绪数据的完整管道。系统输出面向下游 RAG 系统消费（结构化文本块 + 向量），而非人类直接阅读。

当前状态：空项目，仅有 README 标题。需要从零构建四个核心模块：爬虫引擎、文档处理器、向量存储、查询接口。

## Goals / Non-Goals

**Goals:**
- 按 URL 列表逐一爬取页面内容（不递归跟踪链接），保留页面内超链接信息
- 将 HTML 转换为 RAG 友好的结构化文本块（保留元数据：来源 URL、标题、层级关系）
- 向量化存储并支持混合检索（向量相似度 + 关键词检索）
- 提供 CLI 和 REST API 两种交互方式
- 输出格式标准化，便于任意 RAG 系统直接消费

**Non-Goals:**
- 不构建前端 UI（仅 CLI + API）
- 不实现通用搜索引擎功能（仅针对指定站点/URL 集合）
- 不处理需要 JavaScript 渲染的 SPA 页面（v1 仅支持静态 HTML，后续可扩展）
- 不实现用户认证/多租户（v1 为单用户工具）

## Decisions

### 1. 语言与框架：Python
- **选择**：Python 3.11+
- **理由**：NLP/ML 生态最成熟，crawl4ai/scrapy/beautifulsoup 等爬虫库丰富，LangChain/LlamaIndex 集成方便
- **替代方案**：Node.js（爬虫生态弱于 Python）、Go（ML 库生态不足）

### 2. 爬虫引擎：crawl4ai
- **选择**：crawl4ai 作为核心爬取引擎
- **理由**：原生支持 LLM 友好输出，内置 Markdown 转换，async 支持好
- **替代方案**：Scrapy（更重，学习曲线陡）、requests + BeautifulSoup（需手动处理太多）

### 3. 向量数据库：Milvus
- **选择**：Milvus 作为向量存储后端
- **理由**：原生支持混合检索（向量 + 关键词/标量过滤），高性能，可扩展，社区活跃
- **替代方案**：ChromaDB（不支持关键词检索）、Qdrant（混合检索能力弱于 Milvus）、Pinecone（SaaS 依赖）

### 4. Embedding 模型：可插拔，默认 sentence-transformers
- **选择**：默认使用 sentence-transformers（all-MiniLM-L6-v2），同时支持 OpenAI embedding API
- **理由**：本地运行零成本，无 API 依赖；OpenAI 作为可选高质量方案
- **替代方案**：仅 OpenAI（有 API 成本和网络依赖）

### 5. 分块策略：基于语义的递归分割
- **选择**：RecursiveCharacterTextSplitter 风格，按标题/段落/句子层级递归分割
- **理由**：保留文档结构语义，chunk 边界更自然，检索质量高于固定长度分割
- **替代方案**：固定长度分割（简单但破坏语义）、按页分割（粒度过粗）

### 6. 输出格式：JSON Lines + 元数据
- **选择**：每个 chunk 输出为 JSON 对象（含 text、metadata、embedding），整体使用 JSONL 格式
- **理由**：流式处理友好，下游 RAG 系统易解析，保留完整溯源信息
- **替代方案**：纯文本（丢失元数据）、CSV（不适合嵌套结构）

### 7. 项目结构：单包 monolith
- **选择**：单个 Python 包，模块化内部结构
- **理由**：v1 复杂度不需要微服务，部署简单
- **替代方案**：monorepo 多包（过度工程）

## Risks / Trade-offs

- **[静态 HTML 限制]** → v1 不支持 JS 渲染页面。缓解：文档明确标注，后续可集成 Playwright
- **[Milvus 部署复杂度]** → Milvus 需要独立服务进程（或 Milvus Lite 用于开发）。缓解：提供 docker-compose 一键部署，开发环境使用 Milvus Lite
- **[爬取目标站点负载]** → 高并发爬取可能对目标站点造成压力。缓解：默认保守限流（1 req/s），遵守 robots.txt
- **[Embedding 模型质量]** → 本地小模型在特定领域表现可能不如 OpenAI。缓解：提供可插拔接口，用户可按需切换
- **[网络依赖]** → crawl4ai 需要网络访问。缓解：支持本地 HTML 文件输入作为离线模式
