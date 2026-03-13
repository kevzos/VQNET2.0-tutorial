#!/usr/bin/env python
"""
测试新补充的MCP接口
验证QTensor、NN、Optim模块新增接口的代码生成功能
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from simple_mcp_server import SimpleMCPServer

def test_interface(tool_name, arguments):
    """测试单个接口"""
    print(f"\n{'='*60}")
    print(f"测试接口: {tool_name}")
    print(f"参数: {json.dumps(arguments, indent=2, ensure_ascii=False)}")
    print('-'*60)

    server = SimpleMCPServer()

    # 模拟工具调用
    result = server.simulate_tool_call(tool_name, arguments)

    print(f"状态: {result['status']}")
    print(f"消息: {result['message']}")
    print(f"生成代码:\n{result['generated_code']}")

    if 'note' in result:
        print(f"说明: {result['note']}")

    print(f"{'='*60}\n")
    return result

def main():
    """运行新增接口测试"""
    print("测试新补充的MCP接口")
    print(f"测试时间: {os.popen('date +\"%Y-%m-%d %H:%M:%S\"').read().strip()}")

    # 测试QTensor新增接口
    test_cases = [
        # QTensor 创建函数
        ("pyvqnet.tensor.ones", {"shape": "[2, 3]", "dtype": "kfloat32"}),
        ("pyvqnet.tensor.zeros", {"shape": "[2, 3]", "dtype": "kfloat32"}),
        ("pyvqnet.tensor.full", {"shape": "[2, 3]", "value": "5.0"}),
        ("pyvqnet.tensor.eye", {"size": "3"}),
        ("pyvqnet.tensor.arange", {"start": "0", "end": "10", "step": "2"}),
        ("pyvqnet.tensor.linspace", {"start": "0.0", "end": "1.0", "num": "5"}),
        ("pyvqnet.tensor.randn", {"shape": "[2, 3]", "mean": "0.0", "std": "1.0"}),

        # QTensor 数学函数
        ("pyvqnet.tensor.add", {"t1": "tensor1", "t2": "tensor2"}),
        ("pyvqnet.tensor.mul", {"t1": "tensor1", "t2": "tensor2"}),
        ("pyvqnet.tensor.matmul", {"t1": "tensor1", "t2": "tensor2"}),
        ("pyvqnet.tensor.exp", {"t": "tensor1"}),
        ("pyvqnet.tensor.log", {"t": "tensor1"}),
        ("pyvqnet.tensor.sqrt", {"t": "tensor1"}),
        ("pyvqnet.tensor.sin", {"t": "tensor1"}),
        ("pyvqnet.tensor.cos", {"t": "tensor1"}),
        ("pyvqnet.tensor.tan", {"t": "tensor1"}),
        ("pyvqnet.tensor.sums", {"t": "tensor1", "axis": "None"}),
        ("pyvqnet.tensor.mean", {"t": "tensor1", "axis": "None"}),
        ("pyvqnet.tensor.max", {"t": "tensor1", "axis": "None"}),
        ("pyvqnet.tensor.min", {"t": "tensor1", "axis": "None"}),

        # QTensor 操作函数
        ("pyvqnet.tensor.reshape", {"t": "tensor1", "new_shape": "[6, 1]"}),
        ("pyvqnet.tensor.transpose", {"t": "tensor1", "dim": "[1, 0]"}),
        ("pyvqnet.tensor.flatten", {"t": "tensor1", "start": "0", "end": "-1"}),
        ("pyvqnet.tensor.squeeze", {"t": "tensor1", "axis": "-1"}),
        ("pyvqnet.tensor.unsqueeze", {"t": "tensor1", "axis": "0"}),
        ("pyvqnet.tensor.concatenate", {"args": "[tensor1, tensor2]", "axis": "0"}),
        ("pyvqnet.tensor.stack", {"QTensors": "[tensor1, tensor2]", "axis": "0"}),

        # QTensor 比较和逻辑函数
        ("pyvqnet.tensor.equal", {"t1": "tensor1", "t2": "tensor2"}),
        ("pyvqnet.tensor.not_equal", {"t1": "tensor1", "t2": "tensor2"}),
        ("pyvqnet.tensor.greater", {"t1": "tensor1", "t2": "tensor2"}),
        ("pyvqnet.tensor.less", {"t1": "tensor1", "t2": "tensor2"}),
        ("pyvqnet.tensor.logical_and", {"t1": "tensor1", "t2": "tensor2"}),
        ("pyvqnet.tensor.logical_or", {"t1": "tensor1", "t2": "tensor2"}),

        # NN 层
        ("pyvqnet.nn.Linear", {"input_channels": "10", "output_channels": "5"}),
        ("pyvqnet.nn.Conv2d", {"input_channels": "3", "output_channels": "16", "kernel_size": "[3, 3]"}),
        ("pyvqnet.nn.BatchNorm2d", {"channel_num": "16"}),
        ("pyvqnet.nn.LayerNorm1d", {"norm_size": "10"}),
        ("pyvqnet.nn.GroupNorm", {"num_groups": "4", "num_channels": "16"}),
        ("pyvqnet.nn.Embedding", {"num_embeddings": "1000", "embedding_dim": "128"}),
        ("pyvqnet.nn.Dropout", {"dropout_rate": "0.5"}),
        ("pyvqnet.nn.ReLu", {}),
        ("pyvqnet.nn.LeakyReLu", {"alpha": "0.01"}),
        ("pyvqnet.nn.Sigmoid", {}),
        ("pyvqnet.nn.Tanh", {}),
        ("pyvqnet.nn.Softmax", {"axis": "-1"}),

        # NN 损失函数
        ("pyvqnet.nn.CrossEntropyLoss", {}),
        ("pyvqnet.nn.NLL_Loss", {}),
        ("pyvqnet.nn.MSELoss", {}),
        ("pyvqnet.nn.BCELoss", {}),

        # Optim 优化器
        ("pyvqnet.optim.Adam", {"params": "model.parameters()", "lr": "0.001"}),
        ("pyvqnet.optim.SGD", {"params": "model.parameters()", "lr": "0.01", "momentum": "0.9"}),
        ("pyvqnet.optim.Adagrad", {"params": "model.parameters()", "lr": "0.01"}),
        ("pyvqnet.optim.RMSProp", {"params": "model.parameters()", "lr": "0.01", "alpha": "0.99"}),
    ]

    for tool_name, arguments in test_cases:
        try:
            test_interface(tool_name, arguments)
        except Exception as e:
            print(f"Error testing {tool_name}: {e}")
            continue

    print("所有新增接口测试完成！")

if __name__ == "__main__":
    main()