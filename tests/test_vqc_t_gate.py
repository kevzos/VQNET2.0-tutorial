#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试VQC T门实现（函数版t()和类版T()）
"""

import sys
sys.path.insert(0, '.')
from simple_mcp_server import SimpleMCPServer

server = SimpleMCPServer()

print("Testing functional t() gate:")
result = server.simulate_tool_call("pyvqnet.qnn.vqc.t", {"q_machine": "qm", "wires": 1})
print(result["message"])
print(f"代码包含t()调用: {'t(q_machine=qm, wires=1' in result['generated_code']}")
print(f"导入t函数: {'from pyvqnet.qnn.vqc import t' in result['generated_code']}")
print("\n生成代码:")
print(result["generated_code"])
print("\n" + "="*60 + "\n")

print("Testing functional t() gate with dagger:")
result = server.simulate_tool_call("pyvqnet.qnn.vqc.t", {"q_machine": "device", "wires": 0, "use_dagger": True})
print(result["message"])
print(f"use_dagger=True: {'use_dagger=True' in result['generated_code']}")
print("\n生成代码:")
print(result["generated_code"])
print("\n" + "="*60 + "\n")

print("Testing class T() gate:")
result = server.simulate_tool_call("t", {"wires": 2, "trainable": True})
print(result["message"])
print(f"代码包含T类实例化: {'T(wires=2' in result['generated_code']}")
print(f"导入T类: {'from pyvqnet.qnn.vqc import T' in result['generated_code']}")
print(f"可训练: {'trainable=True' in result['generated_code']}")
print("\n生成代码:")
print(result["generated_code"])

print("\nT门测试完成!")
