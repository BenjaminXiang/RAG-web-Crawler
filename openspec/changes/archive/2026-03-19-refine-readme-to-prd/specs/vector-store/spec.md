## ADDED Requirements

### Requirement: 向量化存储
系统 SHALL 将文本块通过 embedding 模型转换为向量，并持久化存储到向量数据库中。

#### Scenario: 批量向量化写入
- **WHEN** 文档处理管道输出一批文本块
- **THEN** 系统批量计算 embedding 并写入向量数据库，每个向量关联其文本块和元数据

#### Scenario: 增量更新
- **WHEN** 同一 URL 的内容被重新爬取
- **THEN** 系统删除该 URL 的旧向量记录，写入新向量记录

### Requirement: JSONL 文件导出
系统 SHALL 支持将文本块及其向量导出为 JSONL 文件格式，每行一个 JSON 对象，包含 text、metadata 和 embedding 字段。

#### Scenario: 导出完整数据
- **WHEN** 用户请求导出爬取结果
- **THEN** 系统生成 JSONL 文件，每行包含 `{"text": "...", "metadata": {...}, "embedding": [...]}`

#### Scenario: 仅导出不含向量
- **WHEN** 用户指定导出时不包含 embedding
- **THEN** 系统生成 JSONL 文件，每行仅包含 `{"text": "...", "metadata": {...}}`

### Requirement: 可插拔 Embedding 模型
系统 SHALL 支持可配置的 embedding 模型，默认使用 sentence-transformers（all-MiniLM-L6-v2），同时支持 OpenAI embedding API。

#### Scenario: 使用本地模型
- **WHEN** 用户未配置 OpenAI API key 或显式选择本地模型
- **THEN** 系统使用 sentence-transformers 本地模型计算 embedding

#### Scenario: 使用 OpenAI 模型
- **WHEN** 用户配置了 OpenAI API key 并选择 OpenAI embedding
- **THEN** 系统通过 OpenAI API 计算 embedding

### Requirement: 相似度检索
系统 SHALL 支持混合检索模式，同时使用向量相似度和关键词匹配，返回 top-K 最相关的文本块及其元数据。

#### Scenario: 混合检索
- **WHEN** 用户输入一个查询文本和 top-K 参数
- **THEN** 系统同时执行向量相似度检索和关键词检索，融合排序后返回 top-K 个文本块

#### Scenario: 仅向量检索
- **WHEN** 用户显式指定仅使用向量检索模式
- **THEN** 系统仅基于向量相似度返回结果

#### Scenario: 仅关键词检索
- **WHEN** 用户显式指定仅使用关键词检索模式
- **THEN** 系统仅基于关键词匹配（BM25）返回结果

#### Scenario: 带过滤条件的检索
- **WHEN** 用户指定按来源 URL 或爬取时间范围过滤
- **THEN** 系统仅在匹配过滤条件的文本块中进行检索

### Requirement: 向量数据库后端
系统 SHALL 使用 Milvus 作为向量数据库后端，支持向量索引和标量字段索引以实现混合检索。

#### Scenario: 使用 Milvus
- **WHEN** 系统启动
- **THEN** 系统连接到配置的 Milvus 实例（开发环境可使用 Milvus Lite）

#### Scenario: 数据持久化
- **WHEN** 系统重启后
- **THEN** 之前存储的向量数据仍然可访问
