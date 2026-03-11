#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试VQC i()函数版恒等门实现
"""

import sys
sys.path.insert(0, '.')
from simple_mcp_server import SimpleMCPServer

server = SimpleMCPServer()

print("Testing functional i() gate:")
result = server.simulate_tool_call("pyvqnet.qnn.vqc.i", {"q_machine": "qm", "wires": 0})
print(result["message"])
print(f"代码包含i()调用: {'i(q_machine=qm, wires=0' in result['generated_code']}")
print(f"导入i函数: {'from pyvqnet.qnn.vqc import i' in result['generated_code']}")
print("\n生成代码:")
print(result["generated_code"])
print("\n" + "="*60 + "\n")

print("Testing functional i() gate with custom params:")
result = server.simulate_tool_call("pyvqnet.qnn.vqc.i", {"q_machine": "device", "wires": 2, "use_dagger": True})
print(result["message"])
print(f"使用wires=2: {'wires=2' in result['generated_code']}")
print(f"use_dagger=True: {'use_dagger=True' in result['generated_code']}")
print("\n生成代码:")
print(result["generated_code"])

print("\nIdentity gate功能测试完成!")
