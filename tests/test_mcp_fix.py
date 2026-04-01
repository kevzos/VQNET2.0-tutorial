#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 MCP 工具调用修复

验证:
1. instructions 字段是否正确返回
2. _meta 字段是否正确设置
3. 核心工具是否设置了 alwaysLoad
"""

import sys
import os
import io

# 设置标准输出编码为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vqnet_mcp_server.server import SimpleMCPServer
import json

def test_initialize_with_instructions():
    """测试初始化响应是否包含 instructions"""
    server = SimpleMCPServer()

    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    }

    response = server.handle_request(request)

    print("=" * 60)
    print("测试 1: Initialize 响应包含 instructions")
    print("=" * 60)

    assert "result" in response, "响应应包含 result"
    result = response["result"]

    assert "instructions" in result, "结果应包含 instructions 字段"
    instructions = result["instructions"]

    print(f"✓ instructions 字段存在")
    print(f"  长度: {len(instructions)} 字符")
    print(f"  前200字符: {instructions[:200]}...")

    # 验证关键内容
    assert "VQNET" in instructions, "应包含 VQNET"
    assert "quantum" in instructions.lower(), "应包含 quantum"
    assert "MCP TOOLS" in instructions, "应提示使用 MCP TOOLS"

    print("✓ instructions 内容验证通过")

    return True

def test_tools_list_with_meta():
    """测试工具列表是否包含 _meta 字段"""
    server = SimpleMCPServer()

    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    }

    response = server.handle_request(request)

    print("\n" + "=" * 60)
    print("测试 2: 工具列表包含 _meta 字段")
    print("=" * 60)

    assert "result" in response, "响应应包含 result"
    result = response["result"]

    assert "tools" in result, "结果应包含 tools 字段"
    tools = result["tools"]

    print(f"✓ 工具数量: {len(tools)}")

    # 检查核心工具是否设置了 alwaysLoad
    always_load_tools = [
        "pyvqnet.tensor.tensor.QTensor",
        "pyvqnet.tensor.ones",
        "pyvqnet.tensor.zeros",
        "pyvqnet.qnn.pq3.quantumlayer.QpandaQProgVQCLayer",
    ]

    found_always_load = 0
    for tool in tools:
        if tool["name"] in always_load_tools:
            if "_meta" in tool and tool["_meta"].get("anthropic/alwaysLoad"):
                found_always_load += 1
                print(f"  ✓ {tool['name']}: alwaysLoad = True")
                assert "anthropic/searchHint" in tool["_meta"], "应有 searchHint"

    print(f"✓ 找到 {found_always_load} 个设置了 alwaysLoad 的核心工具")

    # 检查量子工具是否设置了 searchHint
    quantum_tools_with_hint = 0
    for tool in tools:
        if "qnn" in tool["name"] or "vqc" in tool["name"] or "quantum" in tool["name"].lower():
            if "_meta" in tool and "anthropic/searchHint" in tool["_meta"]:
                quantum_tools_with_hint += 1

    print(f"✓ {quantum_tools_with_hint} 个量子工具设置了 searchHint")

    return True

def test_specific_tool_meta():
    """测试特定工具的 _meta 设置"""
    server = SimpleMCPServer()

    request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/list",
        "params": {}
    }

    response = server.handle_request(request)
    tools = response["result"]["tools"]

    print("\n" + "=" * 60)
    print("测试 3: 特定工具 _meta 设置")
    print("=" * 60)

    # 测试 QTensor
    qtensor = next((t for t in tools if t["name"] == "pyvqnet.tensor.tensor.QTensor"), None)
    assert qtensor is not None, "QTensor 工具应存在"
    assert "_meta" in qtensor, "QTensor 应有 _meta"
    assert qtensor["_meta"]["anthropic/alwaysLoad"] == True, "QTensor 应设置 alwaysLoad"
    print(f"✓ QTensor: _meta = {json.dumps(qtensor['_meta'], indent=2)}")

    # 测试 QpandaQProgVQCLayer
    qpanda = next((t for t in tools if "QpandaQProgVQCLayer" in t["name"]), None)
    assert qpanda is not None, "QpandaQProgVQCLayer 工具应存在"
    assert "_meta" in qpanda, "QpandaQProgVQCLayer 应有 _meta"
    assert qpanda["_meta"]["anthropic/alwaysLoad"] == True, "应设置 alwaysLoad"
    print(f"✓ QpandaQProgVQCLayer: _meta = {json.dumps(qpanda['_meta'], indent=2)}")

    return True

def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("VQNET MCP Server 修复验证测试")
    print("修复: Claude 不自动调用 MCP 工具的问题")
    print("=" * 60)

    tests = [
        test_initialize_with_instructions,
        test_tools_list_with_meta,
        test_specific_tool_meta,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            failed += 1
            print(f"✗ 测试失败: {e}")
        except Exception as e:
            failed += 1
            print(f"✗ 测试出错: {e}")

    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)

    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)