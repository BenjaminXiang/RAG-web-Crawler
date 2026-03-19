## ADDED Requirements

### Requirement: URL 列表爬取
系统 SHALL 接受一组 URL 列表作为输入，逐一爬取每个 URL 对应页面的内容。系统 SHALL NOT 自动跟踪页面内的超链接进行递归爬取。

#### Scenario: 从 URL 列表文件爬取
- **WHEN** 用户提供一个包含多个 URL 的文本文件（每行一个 URL）
- **THEN** 系统逐一请求每个 URL 并提取页面内容

#### Scenario: 从命令行参数传入 URL
- **WHEN** 用户通过 CLI 参数直接传入一个或多个 URL
- **THEN** 系统爬取指定的 URL 页面内容

#### Scenario: 无效 URL 处理
- **WHEN** URL 列表中包含格式错误或无法访问的 URL
- **THEN** 系统跳过该 URL，记录错误日志，继续处理剩余 URL

### Requirement: 保留页面内超链接
系统 SHALL 在提取页面内容时保留页面内的超链接信息（URL 和锚文本），作为元数据附加到对应的文本块中。

#### Scenario: 包含超链接的页面
- **WHEN** 爬取的页面包含内部和外部超链接
- **THEN** 提取的内容中保留超链接的 URL 和对应的锚文本

### Requirement: 速率限制
系统 SHALL 支持可配置的请求速率限制，默认不超过 1 请求/秒。

#### Scenario: 默认速率限制
- **WHEN** 用户未指定速率限制参数
- **THEN** 系统以默认 1 请求/秒的速率进行爬取

#### Scenario: 自定义速率限制
- **WHEN** 用户指定速率限制为 N 请求/秒
- **THEN** 系统以 N 请求/秒的速率进行爬取

### Requirement: robots.txt 遵守
系统 SHALL 在爬取前检查目标站点的 robots.txt，并遵守其中的 Disallow 规则。

#### Scenario: 被 robots.txt 禁止的页面
- **WHEN** 目标 URL 被该站点的 robots.txt Disallow 规则覆盖
- **THEN** 系统跳过该 URL 并记录日志

### Requirement: 并发爬取
系统 SHALL 支持可配置的并发数，默认单线程顺序爬取。

#### Scenario: 并发爬取多个 URL
- **WHEN** 用户配置并发数为 N
- **THEN** 系统同时最多发起 N 个并发请求

### Requirement: 自定义请求头
系统 SHALL 支持自定义 HTTP 请求头（如 User-Agent）。

#### Scenario: 自定义 User-Agent
- **WHEN** 用户配置了自定义 User-Agent
- **THEN** 系统在所有 HTTP 请求中使用该 User-Agent
