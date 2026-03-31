# VQNET MCP Server

VQNET Quantum Machine Learning MCP Server - Provides 318+ VQNET 2.0 API interfaces via Model Context Protocol for Claude Code.

## Installation

```bash
pip install vqnet-mcp-server
```

## Setup for Claude Code

After installation, run the setup command to add the MCP to Claude Code global settings:

```bash
vqnet-mcp setup
```

This will add `vqnet-mcp` to your `~/.claude/settings.json`, making the VQNET tools available in any directory when using Claude Code.

## Remove from Global Settings

```bash
vqnet-mcp remove
```

## Usage in Claude Code

After running `vqnet-mcp setup`, you can use VQNET tools in any conversation with Claude Code:

- Ask Claude to use VQNET quantum computing APIs
- Generate code for quantum neural networks
- Use VQNET tensor operations
- Create quantum circuits and layers

Example prompts:
- "Use vqnet-mcp to create a QTensor"
- "Generate code for a quantum layer with pyvqnet"
- "Show me how to use QMachine for variational quantum computing"

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

## License

MIT License