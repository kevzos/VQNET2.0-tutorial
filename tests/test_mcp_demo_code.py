"""
VQNET MCP Demo Code Examples
用于测试 VQNET MCP Server 的示例代码

在 Claude Code 中，你可以使用这些示例来测试 MCP 工具：
直接问 Claude: "使用 vqnet-mcp 工具帮我生成 XXX 的代码"
"""

# ============================================
# 1. QTensor 基础张量操作测试
# ============================================

# 示例 1: 创建 QTensor
"""
Ask Claude: "使用 pyvqnet.tensor.tensor.QTensor 创建一个 2x3 的张量"
"""

from pyvqnet.tensor import QTensor
import numpy as np

# 创建张量
t1 = QTensor(np.ones([2, 3]))
t2 = QTensor([2, 3, 4, 5], requires_grad=True)
print("张量 t1:", t1)
print("张量 t2:", t2)

# ============================================
# 2. 张量数学运算测试
# ============================================

# 示例 2: 基本运算
"""
Ask Claude: "使用 pyvqnet.tensor add, sub, mul 计算张量运算"
"""

from pyvqnet.tensor import QTensor, add, sub, mul

a = QTensor([[1, 2], [3, 4]])
b = QTensor([[5, 6], [7, 8]])

result_add = add(a, b)
result_sub = sub(a, b)
result_mul = mul(a, b)

print("加法结果:", result_add)
print("减法结果:", result_sub)
print("乘法结果:", result_mul)

# ============================================
# 3. 神经网络层测试 (nn)
# ============================================

# 示例 3: Linear 层
"""
Ask Claude: "使用 pyvqnet.nn.Linear 创建一个线性层并演示前向传播"
"""

from pyvqnet.nn import Linear
from pyvqnet.tensor import QTensor

# 创建线性层: 输入 4 维, 输出 2 维
linear = Linear(4, 2)

# 创建输入数据
x = QTensor([[1.0, 2.0, 3.0, 4.0]])

# 前向传播
output = linear(x)
print("Linear 输出:", output)

# ============================================
# 4. 卷积层测试
# ============================================

# 示例 4: Conv2D
"""
Ask Claude: "使用 pyvqnet.nn.Conv2D 创建卷积层并处理图像数据"
"""

from pyvqnet.nn import Conv2D
from pyvqnet.tensor import QTensor

# 创建卷积层: 输入通道 1, 输出通道 3, 卷积核大小 3x3
conv = Conv2D(1, 3, (3, 3))

# 创建模拟图像数据 (batch=1, channels=1, height=5, width=5)
x = QTensor(np.random.rand(1, 1, 5, 5))

# 前向传播
output = conv(x)
print("Conv2D 输出形状:", output.shape)

# ============================================
# 5. 量子神经网络测试 (qnn.pq3)
# ============================================

# 示例 5: QuantumLayer
"""
Ask Claude: "使用 pyvqnet.qnn.pq3.quantumlayer.QuantumLayer 创建量子神经网络层"
"""

from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
from pyvqnet.tensor import QTensor

# 定义量子电路函数
def quantum_circuit(input_data, weights, machine):
    """简单的量子电路"""
    machine.hadamard(0)
    machine.rx(0, input_data[0])
    machine.ry(0, weights[0])
    return machine.probability()

# 创建量子层
qlayer = QuantumLayer(quantum_circuit, 1, "cpu", 1)

# 输入数据
x = QTensor([0.5])

# 前向传播
output = qlayer(x)
print("量子层输出:", output)

# ============================================
# 6. VQC 变分量子电路测试
# ============================================

# 示例 6: QMachine 和量子门
"""
Ask Claude: "使用 pyvqnet.vqc.QMachine 创建量子电路并应用 Hadamard 和 RX 门"
"""

from pyvqnet.vqc import QMachine, Hadamard, RX, Probability

# 创建量子机器
machine = QMachine(2, "cpu")

# 应用量子门
Hadamard(machine, 0)
RX(machine, 0, 0.5)
Hadamard(machine, 1)

# 测量概率
prob = Probability(machine)
print("量子态概率:", prob)

# ============================================
# 7. 优化器测试
# ============================================

# 示例 7: Adam 优化器
"""
Ask Claude: "使用 pyvqnet.optim.adam.Adam 优化器训练简单神经网络"
"""

from pyvqnet.nn import Linear
from pyvqnet.optim import Adam
from pyvqnet.tensor import QTensor
import numpy as np

# 创建简单网络
model = Linear(2, 1)

# 创建优化器
optimizer = Adam(model.parameters(), lr=0.01)

# 训练数据
x_train = QTensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
y_train = QTensor([[3.0], [5.0], [7.0]])

# 训练循环
for epoch in range(100):
    # 前向传播
    pred = model(x_train)
    loss = ((pred - y_train) ** 2).mean()

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# ============================================
# 8. Torch 后端测试
# ============================================

# 示例 8: 使用 Torch 后端
"""
Ask Claude: "使用 pyvqnet.nn.torch 模块创建兼容 PyTorch 的网络层"
"""

# 注意: 需要先安装 PyTorch
# from pyvqnet.nn.torch import TorchLinear, TorchConv2D

# ============================================
# 测试 MCP 工具调用示例
# ============================================

# 在 Claude Code 中使用这些提示词测试 MCP:
#
# 1. "使用 pyvqnet.tensor.zeros 创建一个 3x3 的零张量"
# 2. "使用 pyvqnet.tensor.matmul 计算两个矩阵的乘积"
# 3. "使用 pyvqnet.nn.ReLu 激活函数处理张量"
# 4. "使用 pyvqnet.nn.CrossEntropyLoss 计算交叉熵损失"
# 5. "使用 pyvqnet.qnn.pq3.template.HardwareEfficientAnsatz 创建量子模板"
# 6. "使用 pyvqnet.vqc.CNOT 实现受控非门"
# 7. "使用 pyvqnet.vqc.MeasureAll 测量所有量子比特"
# 8. "使用 pyvqnet.optim.SGD 优化器更新参数"

print("=" * 50)
print("VQNET MCP Demo Code - 测试完成")
print("=" * 50)