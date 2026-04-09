# VQNET 2.0 MCP Server 项目

## 项目目标
从 `source/rst` 中的 API 文档生成 MCP 工具，让 Claude Code 能自动调用相关工具生成代码。

## 使用方式
当用户输入 VQNET 相关需求时，Claude Code 会调用 MCP 工具：

- **ToolSearchTool**: 输入关键词（如"adam优化器"、"Hadamard门"），返回最匹配的 API 工具及其示例代码
- **直接调用工具**: 如 `pyvqnet.tensor.ones`、`pyvqnet.qnn.vqc.Hadamard` 等

## 核心文件
- `rst_to_mcp.py`: 从 RST 文档解析生成工具定义 JSON
- `vqnet_mcp_server/server.py`: MCP 服务器实现
- `vqnet_mcp_server/vqnet_all_tools.json`: 工具定义文件（320+ API）

## 开发规范

### 测试环境
- Conda 环境: `mini311`

### Git 提交规范
- 提交信息格式: `[feat]` / `[ref]` / `[fix]` + 英文描述
- 只提交必要代码文件，不提交 markdown 文件

### 测试代码
- 测试文件放在 `tests/` 目录

### 文档维护
- RST 中理解不清晰的接口记录到 `TODO.md`
- 每 10 轮对话总结当前状态（时间戳命名）

## MCP 工具分类
- `pyvqnet.tensor`: 张量操作 (QTensor, ones, zeros, arange, randn 等)
- `pyvqnet.qnn.pq3`: pyQPanda3 量子层 (QpandaQProgVQCLayer 等)
- `pyvqnet.qnn.vqc`: 变分量子电路 (Hadamard, RX, RY, RZ, CNOT 等)
- `pyvqnet.nn`: 神经网络层 (Linear, Conv2D, BatchNorm 等)
- `pyvqnet.optim`: 优化器 (Adam, SGD, RMSProp 等)
- `pyvqnet.utils`: 工具函数