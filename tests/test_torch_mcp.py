#!/usr/bin/env python
"""
测试torch API接口的MCP工具实现
"""

import sys
import json

# 20个代表性torch API接口
test_tools = [
    # 后端配置
    "pyvqnet.backends.set_backend",
    "pyvqnet.backends.get_backend",
    # 基类
    "pyvqnet.nn.torch.TorchModule",
    # 神经网络层
    "pyvqnet.nn.torch.Linear",
    "pyvqnet.nn.torch.Conv2D",
    "pyvqnet.nn.torch.Dropout",
    "pyvqnet.nn.torch.BatchNorm2d",
    # 激活函数
    "pyvqnet.nn.torch.ReLu",
    "pyvqnet.nn.torch.Sigmoid",
    "pyvqnet.nn.torch.Softmax",
    "pyvqnet.nn.torch.Tanh",
    "pyvqnet.nn.torch.Gelu",
    "pyvqnet.nn.torch.Softplus",
    # 损失函数
    "pyvqnet.nn.torch.CrossEntropyLoss",
    "pyvqnet.nn.torch.MeanSquaredError",
    # VQC相关
    "pyvqnet.qnn.vqc.torch.QMachine",
    "pyvqnet.qnn.vqc.torch.vqc_basis_embedding",
    "pyvqnet.qnn.vqc.torch.vqc_angle_embedding",
    "pyvqnet.qnn.vqc.torch.RX",
    "pyvqnet.qnn.vqc.torch.Hadamard"
]

def test_tool_call(tool_name, arguments=None):
    """模拟工具调用"""
    print(f"\n{'='*60}")
    print(f"测试工具: {tool_name}")
    print(f"{'='*60}")

    # 导入服务器
    sys.path.insert(0, '.')
    from simple_mcp_server import SimpleMCPServer

    server = SimpleMCPServer()

    # 构造请求
    request = {
        "method": "tools/call",
        "id": 1,
        "params": {
            "name": tool_name,
            "arguments": arguments or {}
        }
    }

    # 处理请求
    response = server.handle_request(request)

    if "error" in response:
        print(f"错误: {response['error']['message']}")
        return False

    result = json.loads(response['result']['content'][0]['text'])
    print(f"状态: {result['status']}")
    print(f"消息: {result['message']}")
    if 'generated_code' in result:
        print("\n生成的代码:")
        print(result['generated_code'])
    if 'note' in result:
        print(f"\n说明: {result['note']}")

    return True

def main():
    """主测试函数"""
    print("开始测试torch API MCP接口实现...")

    success_count = 0
    total_count = len(test_tools)

    # 测试无参数工具
    for tool in test_tools[:15]:  # 前15个不需要额外参数
        if test_tool_call(tool):
            success_count += 1

    # 测试带参数的工具
    if test_tool_call("pyvqnet.backends.set_backend", {"backend_name": "torch"}):
        success_count += 1
    if test_tool_call("pyvqnet.nn.torch.Linear", {"input_channels": 784, "output_channels": 10}):
        success_count += 1
    if test_tool_call("pyvqnet.nn.torch.Conv2D", {"input_channels": 1, "output_channels": 32, "kernel_size": [3, 3]}):
        success_count += 1
    if test_tool_call("pyvqnet.qnn.vqc.torch.QMachine", {"qubit_num": 4}):
        success_count += 1
    if test_tool_call("pyvqnet.qnn.vqc.torch.vqc_basis_embedding", {"basis_state": [1, 0, 1, 1], "q_machine": {"type": "QMachine"}}):
        success_count += 1

    print(f"\n{'='*60}")
    print(f"测试完成: {success_count}/{total_count} 个工具成功")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()