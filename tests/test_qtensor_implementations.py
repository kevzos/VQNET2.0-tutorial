#!/usr/bin/env python
"""
直接测试QTensor接口的实现逻辑，不依赖JSON工具定义
"""

import sys
import json

def test_implementations():
    """测试所有QTensor方法实现"""
    print("开始测试QTensor方法实现...\n")

    # 导入服务器
    sys.path.insert(0, '.')
    from simple_mcp_server import SimpleMCPServer

    server = SimpleMCPServer()

    # 测试用例列表
    test_cases = [
        # (方法名, 参数, 预期包含字符串)
        ("is_dense", {}, "是否是稠密张量"),
        ("is_contiguous", {}, "是否连续存储"),
        ("zero_grad", {}, "将梯度清零"),
        ("backward", {}, "反向传播计算梯度"),
        ("to_numpy", {}, "转换为numpy数组"),
        ("item", {}, "单元素张量的值"),
        ("contiguous", {}, "转换为连续存储"),
        ("argmax", {"dim": 0, "keepdims": True}, "最大值索引"),
        ("argmin", {"dim": 1, "keepdims": False}, "最小值索引"),
        ("fill_", {"value": 42}, "填充后的张量"),
        ("all", {}, "是否所有元素都非零"),
        ("any", {}, "是否有非零元素"),
        ("fill_rand_binary_", {"v": 0.5}, "二项分布填充后的张量"),
        ("fill_rand_signed_uniform_", {"v": 10}, "有符号均匀分布填充后的张量"),
        ("fill_rand_uniform_", {"v": 100}, "均匀分布填充后的张量"),
        ("fill_rand_normal_", {"m": 0, "s": 1, "fast_math": True}, "正态分布填充后的张量"),
    ]

    success_count = 0
    total_count = len(test_cases)

    for method_name, params, expected_str in test_cases:
        print(f"测试 {method_name}...", end=" ")
        try:
            result = server.simulate_tool_call(f"pyvqnet.tensor.tensor.QTensor.{method_name}", params)
            if result["status"] == "success" and expected_str in result["generated_code"]:
                print("✅ 成功")
                success_count += 1
            else:
                print("❌ 失败")
                print(f"   错误: 生成的代码不包含预期字符串 '{expected_str}'")
        except Exception as e:
            print("❌ 异常")
            print(f"   异常信息: {str(e)}")

    print(f"\n{'='*60}")
    print(f"测试完成: {success_count}/{total_count} 个实现成功")
    print(f"{'='*60}")

    return success_count == total_count

if __name__ == "__main__":
    success = test_implementations()
    sys.exit(0 if success else 1)