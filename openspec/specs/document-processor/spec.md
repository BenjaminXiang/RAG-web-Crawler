## ADDED Requirements

### Requirement: Markdown 文件输出
系统 SHALL 将每个爬取页面转换后的 Markdown 内容保存为独立文件，作为重要的中间输出产物。

#### Scenario: 保存 Markdown 文件
- **WHEN** 一个页面被成功爬取并转换为 Markdown
- **THEN** 系统将 Markdown 内容保存到指定输出目录，文件名基于页面 URL 或标题生成

#### Scenario: Markdown 文件包含元数据头
- **WHEN** Markdown 文件被保存
- **THEN** 文件头部包含 YAML front matter，记录来源 URL、页面标题、爬取时间等元数据

#### Scenario: 自定义输出目录
- **WHEN** 用户指定 Markdown 输出目录
- **THEN** 系统将所有 Markdown 文件保存到该目录

### Requirement: HTML 到文本转换
系统 SHALL 将爬取的原始 HTML 转换为清洁的纯文本/Markdown 格式，去除脚本、样式、导航等非内容元素。

#### Scenario: 标准 HTML 页面
- **WHEN** 输入一个包含正文、导航栏、页脚、脚本标签的 HTML 页面
- **THEN** 输出仅包含正文内容的清洁文本，保留标题层级结构

#### Scenario: 空内容页面
- **WHEN** 输入的 HTML 页面正文内容为空或仅包含非文本元素
- **THEN** 系统跳过该页面并记录警告日志

### Requirement: 文本分块
系统 SHALL 将清洁文本按语义边界分割为适合 RAG 消费的文本块（chunks），默认块大小 512 tokens，重叠 50 tokens。

#### Scenario: 长文档分块
- **WHEN** 输入文本长度超过单个块的最大 token 数
- **THEN** 系统按标题/段落/句子层级递归分割，生成多个重叠的文本块

#### Scenario: 短文档不分块
- **WHEN** 输入文本长度小于等于单个块的最大 token 数
- **THEN** 系统将整个文本作为一个块输出

#### Scenario: 自定义块大小
- **WHEN** 用户配置了自定义的块大小和重叠大小
- **THEN** 系统使用用户指定的参数进行分块

### Requirement: 元数据附加
系统 SHALL 为每个文本块附加元数据，包括：来源 URL、页面标题、块在文档中的序号、爬取时间戳。

#### Scenario: 正常分块输出
- **WHEN** 一个页面被分割为多个文本块
- **THEN** 每个块都包含完整的元数据（source_url, title, chunk_index, crawled_at）

### Requirement: 超链接保留
系统 SHALL 将页面中的超链接信息（URL + 锚文本）以结构化形式保留在对应文本块的元数据中。

#### Scenario: 文本块中包含链接
- **WHEN** 某个文本块的原始 HTML 中包含超链接
- **THEN** 该块的元数据中包含 links 字段，记录所有链接的 URL 和锚文本

### Requirement: 多格式支持
系统 SHALL 支持处理 HTML 页面中嵌入的表格和列表结构，将其转换为结构化文本。

#### Scenario: 包含表格的页面
- **WHEN** 页面包含 HTML 表格
- **THEN** 表格被转换为可读的文本格式（如 Markdown 表格），保留行列结构

#### Scenario: 包含嵌套列表的页面
- **WHEN** 页面包含嵌套的有序/无序列表
- **THEN** 列表被转换为带缩进层级的文本格式
