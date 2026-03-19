## ADDED Requirements

### Requirement: CLI 查询接口
系统 SHALL 提供命令行查询接口，接受自然语言查询，返回检索到的相关文本块。

#### Scenario: CLI 语义查询
- **WHEN** 用户通过 CLI 输入查询文本
- **THEN** 系统返回 top-K 相关文本块，包含文本内容、来源 URL 和相似度分数

#### Scenario: 指定返回数量
- **WHEN** 用户通过 CLI 参数指定 top-K 值
- **THEN** 系统返回指定数量的结果

### Requirement: REST API 查询接口
系统 SHALL 提供 HTTP REST API 查询端点，接受 JSON 格式的查询请求，返回 JSON 格式的检索结果。

#### Scenario: API 语义查询
- **WHEN** 客户端发送 POST 请求到 `/api/query`，body 包含 `{"query": "...", "top_k": N}`
- **THEN** 系统返回 JSON 响应，包含 top-K 相关文本块及其元数据和相似度分数

#### Scenario: API 健康检查
- **WHEN** 客户端发送 GET 请求到 `/api/health`
- **THEN** 系统返回 200 状态码和服务状态信息

### Requirement: LLM 增强回答（可选）
系统 SHALL 支持可选的 LLM 增强模式，将检索到的文本块作为上下文传给 LLM，生成带引用的自然语言回答。

#### Scenario: LLM 增强查询
- **WHEN** 用户启用 LLM 模式并发起查询
- **THEN** 系统先检索相关文本块，然后调用 LLM 生成基于这些文本块的回答，回答中包含来源引用

#### Scenario: 仅检索模式
- **WHEN** 用户未启用 LLM 模式或未配置 LLM provider
- **THEN** 系统仅返回检索到的原始文本块，不调用 LLM

### Requirement: CLI 爬取命令
系统 SHALL 提供 CLI 命令触发爬取任务。

#### Scenario: 从文件爬取 URL 列表
- **WHEN** 用户执行 `crawl --urls urls.txt`
- **THEN** 系统读取文件中的 URL 列表，执行爬取、处理、向量化全流程

#### Scenario: 从参数爬取
- **WHEN** 用户执行 `crawl --url https://example.com`
- **THEN** 系统爬取指定 URL 并完成全流程处理

### Requirement: REST API 爬取端点
系统 SHALL 提供 HTTP REST API 端点触发爬取任务。

#### Scenario: API 触发爬取
- **WHEN** 客户端发送 POST 请求到 `/api/crawl`，body 包含 `{"urls": [...]}`
- **THEN** 系统异步执行爬取任务，返回任务 ID

#### Scenario: 查询爬取状态
- **WHEN** 客户端发送 GET 请求到 `/api/crawl/{task_id}`
- **THEN** 系统返回该爬取任务的当前状态（pending/running/completed/failed）
