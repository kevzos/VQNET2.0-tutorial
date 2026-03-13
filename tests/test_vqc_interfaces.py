#!/usr/bin/env python
"""
测试新实现的VQC MCP接口
验证每个接口的代码生成功能是否正常
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from simple_mcp_server import SimpleMCPServer

def test_vqc_interface(tool_name, arguments):
    """测试单个VQC接口"""
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
    """运行所有VQC接口测试"""
    print("测试新实现的10个VQC MCP接口")
    print(f"测试时间: {os.popen('date +"%Y-%m-%d %H:%M:%S"').read().strip()}")

    # 1. 测试MeasureAll
    test_vqc_interface("pyvqnet.qnn.vqc.MeasureAll", {
        "obs": "np.array([[1, 0], [0, -1]], dtype=np.complex64)",
        "name": "measure_all_test"
    })

    # 2. 测试cnot
    test_vqc_interface("pyvqnet.qnn.vqc.cnot", {
        "wires": "[0, 1]",
        "use_dagger": False
    })

    # 3. 测试swap
    test_vqc_interface("pyvqnet.qnn.vqc.swap", {
        "wires": "[0, 1]"
    })

    # 4. 测试toffoli
    test_vqc_interface("pyvqnet.qnn.vqc.toffoli", {
        "wires": "[0, 1, 2]"
    })

    # 5. 测试cz
    test_vqc_interface("pyvqnet.qnn.vqc.cz", {
        "wires": "[0, 1]"
    })

    # 6. 测试rx
    test_vqc_interface("pyvqnet.qnn.vqc.rx", {
        "wires": 0,
        "params": "np.pi/2"
    })

    # 7. 测试ry
    test_vqc_interface("pyvqnet.qnn.vqc.ry", {
        "wires": 0,
        "params": "np.pi/4"
    })

    # 8. 测试rz
    test_vqc_interface("pyvqnet.qnn.vqc.rz", {
        "wires": 0,
        "params": "np.pi/3"
    })

    # 9. 测试VQC_HardwareEfficientAnsatz
    test_vqc_interface("pyvqnet.qnn.vqc.VQC_HardwareEfficientAnsatz", {
        "n_qubits": 3,
        "single_rot_gate_list": '["RY", "RZ"]',
        "entangle_gate": '"CNOT"',
        "depth": 2
    })

    # 10. 测试VQC_AngleEmbedding
    test_vqc_interface("pyvqnet.qnn.vqc.VQC_AngleEmbedding", {
        "input_feat": "[0.1, 0.2, 0.3]",
        "wires": "[0, 1, 2]",
        "rotation": '"Y"'
    })

    print("所有VQC接口测试完成！")

if __name__ == "__main__":
    main()
