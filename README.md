# VQNET MCP Server

VQNET Quantum Machine Learning MCP Server - Provides 318+ VQNET 2.0 API interfaces via Model Context Protocol for Claude Code.

## What's New in v0.3.0

**Fixed: Claude now automatically uses VQNET MCP tools instead of Web Search**

This version includes critical fixes to ensure Claude automatically discovers and uses VQNET tools:

1. **Server Instructions** - Added comprehensive instructions that Claude sees in its system prompt, explaining when to use VQNET tools
2. **Core Tools Pre-loaded** - Essential tools (QTensor, ones, zeros, QpandaQProgVQCLayer, etc.) are now always visible to Claude
3. **Search Hints** - All 318 tools include search hints for better discoverability via ToolSearchTool

## Installation & Setup

### Step 1: Install the package
```bash
pip install vqnet_mcp_server-xxx.whl
```

Or from PyPI:
```bash
pip install vqnet-mcp-server
```

### Step 2: Configure Claude Code
```bash
vqnet-mcp setup
```

This adds `vqnet-mcp` to your `~/.claude.json` (user-level settings), making VQNET tools available in any directory.

### Step 3: Restart Claude Code
Restart Claude Code to load the new MCP server.

## Commands

| Command | Description |
|---------|-------------|
| `vqnet-mcp setup` | Add MCP to Claude Code user settings |
| `vqnet-mcp remove` | Remove MCP from user settings |
| `vqnet-mcp --help` | Show help information |
| `vqnet-mcp` | Run MCP server (usually auto-started by Claude Code) |

## Verify Installation

After setup, restart Claude Code and ask about VQNET:
- "What quantum gates are available in vqnet-mcp?"
- "Show me how to create a QTensor"

Claude will automatically use VQNET MCP tools when you ask about quantum computing topics.

## Available APIs (318 tools)

### Tensor Operations
- `QTensor`, `zeros`, `ones`, `full`, `arange`, `linspace`
- `add`, `sub`, `mul`, `divide`, `matmul`
- `exp`, `log`, `sqrt`, `abs`, `power`
- `reshape`, `transpose`, `flatten`, `concatenate`

### Quantum Neural Networks (qnn.pq3)
- `QuantumLayer`, `QpandaQProgVQCLayer`, `QuantumLayerAdjoint`
- `AmplitudeEmbeddingCircuit`, `AngleEmbeddingCircuit`
- `HardwareEfficientAnsatz`, `BasicEntanglerTemplate`

### Variational Quantum Circuits (vqc)
- `QMachine`, `Hadamard`, `PauliX`, `PauliY`, `PauliZ`
- `RX`, `RY`, `RZ`, `CRX`, `CRY`, `CRZ`
- `cnot`, `swap`, `toffoli`, `cz`
- `Probability`, `MeasureAll`, `Samples`, `HermitianExpval`
- `VQC_HardwareEfficientAnsatz`, `VQC_AngleEmbedding`

### Neural Network Layers (nn)
- `Linear`, `Conv2D`, `BatchNorm2d`, `Dropout`
- `ReLu`, `Sigmoid`, `Softmax`, `Tanh`, `Gelu`
- `CrossEntropyLoss`, `MSELoss`

### Optimizers
- `Adam`, `AdamW`, `SGD`, `RMSProp`, `Adagrad`

### Torch Backend
- Full PyTorch-compatible API via `pyvqnet.nn.torch`

## Requirements

- Python >= 3.8
- pyVQNET (install separately: `pip install pyvqnet`)

## Development & Building

### Build wheel package
```bash
# 安装 build 工具
pip install build

# 打包 wheel
python -m build

# 或者只构建 wheel（不包含 sdist）
python -m build --wheel
```

生成的 wheel 文件在 `dist/` 目录下。

### Install from source (editable mode)
```bash
pip install -e .
```

## License

MIT License