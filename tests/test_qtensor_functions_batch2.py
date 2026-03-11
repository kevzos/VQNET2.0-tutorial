#!/usr/bin/env python
"""
测试QTensor第二批全局函数实现
"""

import sys
sys.path.insert(0, '.')
from simple_mcp_server import SimpleMCPServer

server = SimpleMCPServer()

# 测试函数列表
test_functions = [
    ("argsort", {"axis": 1, "descending": False}),
    ("topK", {"k": 5, "axis": -1, "if_descent": True}),
    ("argtopK", {"k": 3, "axis": 1, "if_descent": False}),
    ("add", {}),
    ("sub", {}),
    ("mul", {}),
    ("divide", {}),
    ("sums", {"axis": 0, "keepdims": True}),
    ("cumsum", {"axis": 1}),
    ("mean", {"axis": None}),
    ("median", {"axis": 1, "keepdims": False}),
]

print("测试QTensor第二批全局函数实现:\n")

all_success = True
for func_name, params in test_functions:
    print(f"{'='*60}")
    print(f"测试函数: {func_name}")
    print(f"参数: {params}")
    print(f"{'-'*60}")
    try:
        result = server.simulate_tool_call(func_name, params)
        if result["status"] == "success":
            print(f"✅ 成功: {result['message']}")
            print(f"生成代码示例:\n{result['generated_code'][:200]}...")
        else:
            print(f"❌ 失败: {result.get('message', '未知错误')}")
            all_success = False
    except Exception as e:
        print(f"❌ 异常: {str(e)}")
        import traceback
        traceback.print_exc()
        all_success = False
    print()

print(f"{'='*60}")
print(f"测试结果: {'全部通过' if all_success else '存在失败'}")
print(f"{'='*60}")
