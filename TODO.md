# VQNET MCP 工具待实现/跳过接口列表

## 已跳过的接口（共12个）
以下接口由于需要真实量子硬件、参数复杂或实现难度较高，暂时跳过：

### qnn_pq3.rst 跳过接口
1. `pyvqnet.qnn.pq3.quantumlayer.QuantumBatchAsyncQcloudLayer` - 需要真实量子硬件访问
2. `pyvqnet.qnn.pq3.quantumlayer.QuantumLayerAdjoint` - 依赖VQCircuit特定内部接口
3. `pyvqnet.qnn.pq3.template.FermionicSingleExcitation` - 参数结构复杂，使用频率低
4. `pyvqnet.qnn.pq3.template.FermionicDoubleExcitation` - 参数结构复杂，使用频率低
5. `pyvqnet.qnn.pq3.template.UCCSD` - 高级量子化学拟设，实现复杂
6. `pyvqnet.qnn.pq3.template.QuantumPoolingCircuit` - 池化线路参数复杂

### vqc.rst 跳过接口
7. `pyvqnet.qnn.vqc.ExpressiveEntanglingAnsatz` - 复杂拟设类，实现难度高
8. `pyvqnet.qnn.vqc.VQC_HardwareEfficientAnsatz` - 硬件高效拟设，内部实现复杂
9. `pyvqnet.qnn.vqc.VQC_BasicEntanglerTemplate` - 基础纠缠模板，依赖内部组件
10. `pyvqnet.qnn.vqc.VQC_StronglyEntanglingTemplate` - 强纠缠模板，实现复杂
11. `pyvqnet.qnn.vqc.VQC_QuantumEmbedding` - 量子嵌入类，内部逻辑复杂
12. `pyvqnet.qnn.vqc.single_excitation/double_excitation` - 激发线路数学实现困难

## 待实现的API（共33个，按优先级排序）
### 高优先级（常用基础API）
- QTensor所有属性和方法（ndim、shape、size、dtype等）
- 基础量子门（I、PauliX/Y/Z、T、S、RY/RZ等）
- 基础嵌入线路（BasicEmbedding、RotCircuit等）
- 测量模块（MeasureAll、Samples、expval等）

### 中优先级（扩展功能API）
- 受控门（Controlled_Hadamard、CCZ、CSWAP等）
- 拟设模板（HardwareEfficientAnsatz等）
- 量子测量（QuantumMeasure等）
