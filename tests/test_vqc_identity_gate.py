#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试VQC Identity门实现
"""

import sys
sys.path.insert(0, '.')
from simple_mcp_server import SimpleMCPServer

server = SimpleMCPServer()

print("Testing Identity gate (I):")
result = server.simulate_tool_call("identity", {"wires": 0})
print(result["message"])
print(f"代码包含I门: {'I' in result['generated_code']}")
print(f"导入I门: {'from pyvqnet.qnn.vqc import I' in result['generated_code']}")
print("\n生成代码:")
print(result["generated_code"])
print("\n" + "="*60 + "\n")

print("Testing I gate with custom wire:")
result = server.simulate_tool_call("i", {"wires": 2, "trainable": True})
print(result["message"])
print(f"使用wires=2: {'wires=2' in result['generated_code']}")
print(f"可训练: {'trainable=True' in result['generated_code']}")
print("\n生成代码:")
print(result["generated_code"])

print("\n✅ Identity gate测试完成!")
