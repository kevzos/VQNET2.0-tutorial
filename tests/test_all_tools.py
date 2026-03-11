#!/usr/bin/env python
"""
测试所有VQNET MCP工具的代码生成功能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_mcp_server import SimpleMCPServer

def test_all_tools():
    """测试所有工具的代码生成"""
    server = SimpleMCPServer()

    print(f"加载了 {len(server.tools)} 个工具:")
    for i, tool in enumerate(server.tools):
        print(f"{i+1}. {tool['name']}")

    # 测试用例
    test_cases = [
        # QTensor测试
        {
            "name": "pyvqnet.tensor.tensor.QTensor",
            "arguments": {
                "data": "[1, 2, 3, 4]",
                "requires_grad": "True"
            }
        },
        # QuantumLayer测试
        {
            "name": "pyvqnet.qnn.pq3.quantumlayer.QuantumLayer",
            "arguments": {
                "qprog_with_measure": """def pqctest (input,param):
    num_of_qubits = 4
    m_machine = pq.CPUQVM()
    qubits = range(num_of_qubits)
    circuit = pq.QCircuit()
    circuit<<pq.H(qubits[0])
    circuit<<pq.RZ(qubits[0], input[0])
    circuit<<pq.RZ(qubits[1], param[0])
    prog = pq.QProg()
    prog<<circuit
    rlt_prob = ProbsMeasure(m_machine, prog, [0,1])
    return rlt_prob""",
                "para_num": 1
            }
        },
        # grad函数测试
        {
            "name": "pyvqnet.qnn.pq3.quantumlayer.grad",
            "arguments": {
                "quantum_prog_func": """def pqctest(param):
    machine = pq.CPUQVM()
    qubits = range(2)
    circuit = pq.QCircuit(2)
    circuit<<pq.RX(qubits[0], param[0])
    circuit<<pq.RY(qubits[1], param[1])
    prog = pq.QProg()
    prog<<circuit
    EXP = ProbsMeasure(machine, prog, [1])
    return EXP""",
                "input_params": "[0.1, 0.2]"
            }
        },
        # AmplitudeEmbeddingCircuit测试
        {
            "name": "pyvqnet.qnn.pq3.template.AmplitudeEmbeddingCircuit",
            "arguments": {
                "input_feat": "np.array([0.5, 0.5, 0.5, 0.5])",
                "qubits": "[0, 1]"
            }
        },
        # AngleEmbeddingCircuit测试
        {
            "name": "pyvqnet.qnn.pq3.template.AngleEmbeddingCircuit",
            "arguments": {
                "input_feat": "np.array([0.1, 0.2, 0.3])",
                "qubits": "[0, 1, 2]",
                "rotation": "Y"
            }
        },
        # QMachine测试
        {
            "name": "pyvqnet.qnn.vqc.QMachine",
            "arguments": {
                "num_wires": "4"
            }
        },
        # hadamard门测试
        {
            "name": "pyvqnet.qnn.vqc.hadamard",
            "arguments": {
                "q_machine": "qm",
                "wires": "0"
            }
        },
        # Probability测试
        {
            "name": "pyvqnet.qnn.vqc.Probability",
            "arguments": {
                "wires": "0"
            }
        }
    ]

    print("\n" + "="*80)
    print("开始测试工具代码生成...")
    print("="*80)

    all_passed = True
    for i, test_case in enumerate(test_cases):
        tool_name = test_case["name"]
        arguments = test_case["arguments"]

        print(f"\n测试 {i+1}: {tool_name}")
        print("-"*60)

        try:
            result = server.simulate_tool_call(tool_name, arguments)
            if result.get("status") == "success":
                print("[PASS] 生成成功!")
                print("生成的代码:")
                print(result["generated_code"])
                print("\n备注:", result.get("note", ""))
            else:
                print("[FAIL] 生成失败!")
                print("错误信息:", result.get("message", ""))
                all_passed = False
        except Exception as e:
            print(f"[FAIL] 测试出错: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

        print("-"*60)

    print("\n" + "="*80)
    if all_passed:
        print("[PASS] 所有工具测试通过!")
    else:
        print("[FAIL] 部分工具测试失败!")
    print("="*80)

    return all_passed

if __name__ == "__main__":
    success = test_all_tools()
    sys.exit(0 if success else 1)
