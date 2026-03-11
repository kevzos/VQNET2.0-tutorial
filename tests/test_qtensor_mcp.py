#!/usr/bin/env python
"""
测试QTensor接口的MCP工具实现
"""

import sys
import json

# 测试工具列表
test_tools = [
    "pyvqnet.tensor.tensor.QTensor.is_dense",
    "pyvqnet.tensor.tensor.QTensor.is_contiguous",
    "pyvqnet.tensor.tensor.QTensor.zero_grad",
    "pyvqnet.tensor.tensor.QTensor.backward",
    "pyvqnet.tensor.tensor.QTensor.to_numpy",
    "pyvqnet.tensor.tensor.QTensor.item",
    "pyvqnet.tensor.tensor.QTensor.contiguous",
    "pyvqnet.tensor.tensor.QTensor.argmax",
    "pyvqnet.tensor.tensor.QTensor.argmin",
    "pyvqnet.tensor.tensor.QTensor.fill_",
    "pyvqnet.tensor.tensor.QTensor.all",
    "pyvqnet.tensor.tensor.QTensor.any",
    "pyvqnet.tensor.tensor.QTensor.fill_rand_binary_",
    "pyvqnet.tensor.tensor.QTensor.fill_rand_signed_uniform_",
    "pyvqnet.tensor.tensor.QTensor.fill_rand_uniform_",
    "pyvqnet.tensor.tensor.QTensor.fill_rand_normal_",
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
    print("开始测试QTensor MCP接口实现...")

    success_count = 0
    total_count = len(test_tools)

    # 测试无参数工具
    for tool in test_tools[:12]:  # 前12个不需要额外参数
        if test_tool_call(tool):
            success_count += 1

    # 测试带参数的工具
    if test_tool_call("pyvqnet.tensor.tensor.QTensor.argmax", {"dim": 0, "keepdims": True}):
        success_count +=1
    if test_tool_call("pyvqnet.tensor.tensor.QTensor.argmin", {"dim": 1, "keepdims": False}):
        success_count +=1
    if test_tool_call("pyvqnet.tensor.tensor.QTensor.fill_", {"value": 42}):
        success_count +=1
    if test_tool_call("pyvqnet.tensor.tensor.QTensor.fill_rand_binary_", {"v": 0.5}):
        success_count +=1
    if test_tool_call("pyvqnet.tensor.tensor.QTensor.fill_rand_signed_uniform_", {"v": 10}):
        success_count +=1
    if test_tool_call("pyvqnet.tensor.tensor.QTensor.fill_rand_uniform_", {"v": 100}):
        success_count +=1
    if test_tool_call("pyvqnet.tensor.tensor.QTensor.fill_rand_normal_", {"m": 0, "s": 1, "fast_math": True}):
        success_count +=1

    print(f"\n{'='*60}")
    print(f"测试完成: {success_count}/{total_count} 个工具成功")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()