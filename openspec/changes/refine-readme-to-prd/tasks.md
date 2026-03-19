## 1. 项目初始化

- [x] 1.1 创建 Python 项目结构（pyproject.toml、src 目录、入口模块）
- [x] 1.2 配置 Python 依赖：crawl4ai、pymilvus、sentence-transformers、openai、fastapi、uvicorn
- [x] 1.3 创建配置文件模板（config.yaml），定义爬取参数、Milvus 连接、embedding 模型等配置项
- [x] 1.4 编写 docker-compose.yaml（Milvus + 应用服务）

## 2. Web 爬虫引擎（web-crawler）

- [x] 2.1 实现 URL 列表读取：支持文件输入和 CLI 参数传入
- [x] 2.2 基于 crawl4ai 实现单页面爬取，提取 HTML 内容
- [x] 2.3 实现速率限制（默认 1 req/s）和可配置并发数
- [x] 2.4 实现 robots.txt 检查与遵守
- [x] 2.5 实现自定义 HTTP 请求头支持（User-Agent 等）
- [x] 2.6 实现错误处理：无效 URL 跳过并记录日志
- [ ] 2.7 编写爬虫模块单元测试

## 3. 文档处理器（document-processor）— 核心输出

- [x] 3.1 实现 HTML → Markdown 转换（LLM 增强 + 规则清理双模式）
- [x] 3.2 保留页面内超链接信息（URL + 锚文本），在 Markdown 中以标准链接格式体现
- [x] 3.3 实现表格和嵌套列表的结构化 Markdown 转换（LLM 自动处理）
- [x] 3.4 实现 Markdown 文件输出：每个 URL 生成独立文件夹（有意义命名），文件头包含 YAML front matter
- [x] 3.5 实现链接附件收集：提取页面中的附件链接（PDF、图片等），记录到元数据中
- [x] 3.6 实现文本分块：基于语义递归分割（默认 512 tokens，50 tokens 重叠）
- [x] 3.7 实现空内容/无效页面跳过与日志记录
- [ ] 3.8 编写文档处理模块单元测试

## 4. 向量存储（vector-store）

- [x] 4.1 实现 Milvus 连接管理与 Collection 初始化（向量字段 + 标量字段 + BM25 索引）
- [x] 4.2 实现可插拔 Embedding 接口：默认 sentence-transformers，可选 OpenAI embedding API
- [x] 4.3 实现批量向量化写入：将 chunks 的 embedding 和元数据写入 Milvus
- [x] 4.4 实现增量更新：按 source_url 删除旧记录后写入新记录
- [x] 4.5 实现混合检索：向量相似度 + BM25 关键词检索，支持融合排序
- [x] 4.6 实现纯向量检索和纯关键词检索模式切换
- [x] 4.7 实现标量过滤条件检索（按来源 URL、爬取时间范围）
- [x] 4.8 实现 JSONL 文件导出（含/不含 embedding 两种模式）
- [x] 4.9 编写向量存储模块单元测试

## 5. 查询接口（rag-query）

- [x] 5.1 实现 CLI 查询命令：接受查询文本和 top-K 参数，输出检索结果
- [x] 5.2 实现 CLI 爬取命令：`crawl --urls <file>` 和 `crawl --url <url>` 触发全流程
- [x] 5.3 实现 REST API 查询端点 `POST /api/query`（FastAPI）
- [x] 5.4 实现 REST API 爬取端点 `POST /api/crawl`（异步任务 + 状态查询）
- [x] 5.5 实现 REST API 健康检查 `GET /api/health`
- [x] 5.6 实现可选 LLM 增强回答模式（本地模型 / OpenAI，检索 + 生成带引用回答）
- [x] 5.7 编写 API 接口测试

## 6. 集成与文档

- [x] 6.1 实现端到端流程集成：URL 输入 → 爬取 → Markdown 输出 + 向量化存储
- [ ] 6.2 编写 README.md：项目介绍、快速开始、配置说明、API 文档
- [x] 6.3 编写端到端集成测试
