# VQNET MCP 工具待实现/跳过接口列表

## 已跳过的接口（共4个）
以下接口由于需要真实量子硬件、参数复杂或实现难度较高，暂时跳过：

### qnn_pq3.rst 跳过接口
1. `pyvqnet.qnn.pq3.quantumlayer.QuantumBatchAsyncQcloudLayer` - 需要真实量子硬件访问
2. `pyvqnet.qnn.pq3.template.UCCSD` - 高级量子化学拟设，实现复杂

### vqc.rst 跳过接口
3. `pyvqnet.qnn.vqc.ExpressiveEntanglingAnsatz` - 复杂拟设类，实现难度高
4. `pyvqnet.qnn.vqc.single_excitation/double_excitation` - 激发线路数学实现困难

## 已实现的接口（通过MCP工具生成，共195个）
以下接口已通过rst_to_mcp.py自动生成MCP工具实现：

### QTensor模块（约60个接口）
- 创建函数：ones, zeros, full, eye, arange, linspace, logspace, randu, randn等
- 数学函数：add, sub, mul, divide, exp, log, sqrt, sin, cos, tan等
- 操作函数：reshape, transpose, flatten, squeeze, unsqueeze, concatenate, stack等
- 比较和逻辑函数

### NN模块（约30个接口）
- 网络层：Linear, Conv1d, Conv2d, ConvT2d, BatchNorm1d/2d, LayerNorm1d/2d/Nd, GroupNorm, Embedding, Dropout, DropPath, AvgPool1d/2D, MaxPool1d/2D, GRU, RNN, LSTM, DynamicGRU/RNN/LSTM, Interpolate等
- 激活函数：ReLu, LeakyReLu, Sigmoid, Gelu, ELU, Softplus, Softsign, Tanh, Softmax, HardSigmoid等
- 损失函数：CrossEntropyLoss, NLL_Loss, MSELoss, BCELoss, SoftmaxCrossEntropy等

### Optim模块（8个接口）
- 优化器：Adam, AdamW, Adamax, Adadelta, Adagrad, RMSProp, SGD, Rotosolve

### VQC模块（34个接口）
- 基础量子门（类）：I, Hadamard, T, S, PauliX, PauliY, PauliZ, X1
- 基础量子门（函数）：hadamard, t, s, paulix, pauliy, pauliz, x1
- 参数门（类）：RX, RY, RZ
- 参数门（函数）：rx, ry, rz
- 双量子比特门：cnot, swap, cz, toffoli
- 测量操作：Measure, Probability, Samples, HermitianExpval, MeasureAll, VQC_VarMeasure
- VQC模板：VQC_HardwareEfficientAnsatz, VQC_AngleEmbedding, VQC_RotCircuit
- QMachine, reset_states

### QNN_PQ3模块（28个接口）⭐ 扩展
- 量子层（4个）：QuantumLayer, QpandaQProgVQCLayer, QuantumLayerAdjoint, grad
- 编码线路（6个）：AmplitudeEmbeddingCircuit, AngleEmbeddingCircuit, BasicEmbeddingCircuit, IQPEmbeddingCircuits, RotCircuit, CRotCircuit
- 量子门和模板（12个）：CSWAPcircuit, Controlled_Hadamard, CCZ, HardwareEfficientAnsatz, BasicEntanglerTemplate, StronglyEntanglingTemplate, ComplexEntangelingTemplate, Quantum_Embedding, QuantumPoolingCircuit, FermionicSingleExcitation, FermionicDoubleExcitation
- 测量（6个）：expval, QuantumMeasure, ProbsMeasure, DensityMatrixFromQstate, VN_Entropy, Mutal_Info, Purity

## 待实现的API（按优先级排序）
### 中优先级（扩展功能API）
- 更多NN层（ConvT2d, DynamicGRU等的函数式接口）

### 低优先级
- UCCSD（高级量子化学模板，参数结构复杂）
- 其他内部复杂接口

## 注意事项
- Y1, Z1 门在rst文档中没有独立的类/函数定义，仅在示例代码中出现
- QTensor的ndim、shape、size、dtype等属性是Python对象属性而非函数，无法通过MCP函数式接口直接调用
