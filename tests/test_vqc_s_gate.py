#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试VQC S门实现（函数版s()和类版S()）
"""

import sys
sys.path.insert(0, '.')
from simple_mcp_server import SimpleMCPServer

server = SimpleMCPServer()

print("Testing functional s() gate:")
result = server.simulate_tool_call("pyvqnet.qnn.vqc.s", {"q_machine": "qm", "wires": 1})
print(result["message"])
print(f"代码包含s()调用: {'s(q_machine=qm, wires=1' in result['generated_code']}")
print(f"导入s函数: {'from pyvqnet.qnn.vqc import s' in result['generated_code']}")
print("\n生成代码:")
print(result["generated_code"])
print("\n" + "="*60 + "\n")

print("Testing functional s() gate with dagger:")
result = server.simulate_tool_call("pyvqnet.qnn.vqc.s", {"q_machine": "device", "wires": 0, "use_dagger": True})
print(result["message"])
print(f"use_dagger=True: {'use_dagger=True' in result['generated_code']}")
print("\n生成代码:")
print(result["generated_code"])
print("\n" + "="*60 + "\n")

print("Testing class S() gate:")
result = server.simulate_tool_call("s", {"wires": 2, "trainable": True})
print(result["message"])
print(f"代码包含S类实例化: {'S(wires=2' in result['generated_code']}")
print(f"导入S类: {'from pyvqnet.qnn.vqc import S' in result['generated_code']}")
print(f"可训练: {'trainable=True' in result['generated_code']}")
print("\n生成代码:")
print(result["generated_code"])

print("\nS门测试完成!")
