#!/usr/bin/env python
"""
简单的 MCP stdio 服务器实现
专为 Claude Code 集成设计

实现基本的 MCP 协议，通过 stdio 与 Claude Code 通信。
"""

import json
import sys
import traceback
from typing import Dict, Any, List, Optional
import os

class SimpleMCPServer:
    """简单的 MCP 服务器实现"""

    def __init__(self, tools_file: str = "vqnet_all_tools.json"):
        self.tools_file = tools_file
        self.tools = self.load_tools()

    def load_tools(self) -> List[Dict[str, Any]]:
        """从文件加载工具定义"""
        try:
            with open(self.tools_file, 'r', encoding='utf-8') as f:
                tools = json.load(f)
            print(f"已加载 {len(tools)} 个工具定义", file=sys.stderr)
            return tools
        except FileNotFoundError:
            print(f"错误: 工具文件 {self.tools_file} 未找到", file=sys.stderr)
            print("请先运行 rst_to_mcp.py 生成工具定义", file=sys.stderr)
            return []
        except json.JSONDecodeError as e:
            print(f"错误: 无法解析 JSON 文件: {e}", file=sys.stderr)
            return []

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理 JSON-RPC 请求"""
        method = request.get("method", "")
        request_id = request.get("id")

        # 只记录方法名，不记录完整请求（避免日志过长）
        print(f"处理请求: {method}", file=sys.stderr)

        try:
            if method == "initialize":
                return self.handle_initialize(request, request_id)
            elif method == "tools/list":
                return self.handle_tools_list(request, request_id)
            elif method == "tools/call":
                return self.handle_tools_call(request, request_id)
            elif method == "notifications/initialized":
                # 初始化完成通知，返回空响应
                return {"jsonrpc": "2.0", "id": request_id, "result": None}
            elif method == "shutdown":
                return {"jsonrpc": "2.0", "id": request_id, "result": None}
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"方法不存在: {method}"
                    }
                }
        except Exception as e:
            print(f"处理请求时出错: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"内部错误: {str(e)}"
                }
            }

    def handle_initialize(self, request: Dict[str, Any], request_id: Any) -> Dict[str, Any]:
        """处理初始化请求"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "VQNET Quantum ML Server",
                    "version": "1.0.0"
                }
            }
        }

    def handle_tools_list(self, request: Dict[str, Any], request_id: Any) -> Dict[str, Any]:
        """处理工具列表请求"""
        # 转换工具格式为 MCP 格式
        mcp_tools = []
        for tool in self.tools:
            mcp_tool = {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "inputSchema": tool.get("inputSchema", {})
            }
            mcp_tools.append(mcp_tool)

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": mcp_tools
            }
        }

    def handle_tools_call(self, request: Dict[str, Any], request_id: Any) -> Dict[str, Any]:
        """处理工具调用请求"""
        params = request.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        print(f"调用工具: {tool_name}", file=sys.stderr)
        print(f"参数: {json.dumps(arguments, indent=2, ensure_ascii=False)[:200]}...", file=sys.stderr)

        # 查找工具
        tool_def = None
        for tool in self.tools:
            if tool["name"] == tool_name:
                tool_def = tool
                break

        if not tool_def:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": f"工具未找到: {tool_name}"
                }
            }

        # 模拟工具调用
        result = self.simulate_tool_call(tool_name, arguments)

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, indent=2, ensure_ascii=False)
                    }
                ]
            }
        }

    def simulate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """模拟工具调用"""
        if "QpandaQProgVQCLayer" in tool_name:
            # 生成 Python 代碼片段
            # 提取參數（假設從 arguments 來）
            origin_qprog_func = arguments.get('origin_qprog_func')
            para_num = arguments.get('para_num')
            qvm_type = arguments.get('qvm_type', 'cpu')
            pauli_str_dict = arguments.get('pauli_str_dict', None)
            shots = arguments.get('shots', 1000)
            initializer = arguments.get('initializer', None)
            name = arguments.get('name', "")
            # Format strings properly with quotes
            qvm_type_str = f'"{qvm_type}"' if isinstance(qvm_type, str) else qvm_type
            name_str = f'"{name}"' if isinstance(name, str) else name
            initializer_str = initializer if initializer is not None else 'None'
            pauli_str_dict_str = pauli_str_dict if pauli_str_dict is not None else 'None'


            if origin_qprog_func.startswith("def ") and "(" in origin_qprog_func:
                try:
                    func_name = origin_qprog_func.split("def ", 1)[1].split("(")[0].strip()
                except:
                    func_name = "qfun"  # fallback
            else:
                func_name = origin_qprog_func or "qfun"

            generated_code = f"""
            import pyqpanda3.core as pq
            import pyvqnet
            from pyvqnet.qnn.pq3.quantumlayer import  QpandaQProgVQCLayer

            {origin_qprog_func}
            from pyvqnet.utils.initializer import ones
            l = QpandaQProgVQCLayer({func_name},
                            {para_num},
                            {qvm_type_str},
                            pauli_str_dict={pauli_str_dict_str},
                            shots={shots},
                            initializer={initializer_str},
                            name={name_str})
            """

            return {
                "status": "success",
                "message": f"已生成量子層代碼基於 {tool_name}",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "這是基於 pyVQNet API 生成的真實代碼。確保 origin_qprog_func 已定義並返回 QProg。"
            }
        elif "pyvqnet.tensor.tensor.QTensor" in tool_name:
            # QTensor 类生成
            data = arguments.get('data')
            requires_grad = arguments.get('requires_grad')
            device = arguments.get('device')
            dtype = arguments.get('dtype')

            # 只保留用户提供的参数，默认参数省略
            params = [data]
            if requires_grad is not None:
                requires_grad_str = requires_grad if isinstance(requires_grad, bool) else requires_grad
                params.append(f'requires_grad={requires_grad_str}')
            if device is not None:
                params.append(f'device={device}')
            if dtype is not None:
                params.append(f'dtype={dtype}')

            params_str = ", ".join(params)

            generated_code = f"""
            import numpy as np
            from pyvqnet.tensor import QTensor
            from pyvqnet.dtype import *

            # 创建QTensor实例
            tensor = QTensor({params_str})
            """

            return {
                "status": "success",
                "message": f"已生成QTensor创建代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet QTensor API生成的真实代码。data可以是列表、numpy数组或其他张量。device默认使用pyvqnet.DEV_CPU。"
            }
        elif "pyvqnet.qnn.pq3.quantumlayer.QuantumLayer" in tool_name:
            # QuantumLayer 类生成
            qprog_with_measure = arguments.get('qprog_with_measure')
            para_num = arguments.get('para_num')

            if qprog_with_measure.startswith("def ") and "(" in qprog_with_measure:
                try:
                    func_name = qprog_with_measure.split("def ", 1)[1].split("(")[0].strip()
                except:
                    func_name = "pqctest"  # fallback
            else:
                func_name = qprog_with_measure or "pqctest"

            # 只保留必填参数
            generated_code = f"""
            import pyqpanda3.core as pq
            import numpy as np
            from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
            from pyvqnet.tensor import QTensor, ones

            {qprog_with_measure}

            # 创建QuantumLayer实例
            quantum_layer = QuantumLayer({func_name}, {para_num})
            """

            return {
                "status": "success",
                "message": f"已生成QuantumLayer创建代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet QuantumLayer API生成的真实代码。确保qprog_with_measure函数返回测量结果或期望值。可选参数diff_method、delta等使用默认值。"
            }
        elif "pyvqnet.qnn.pq3.quantumlayer.grad" in tool_name:
            # grad 函数生成
            quantum_prog_func = arguments.get('quantum_prog_func')
            input_params = arguments.get('input_params')

            if quantum_prog_func.startswith("def ") and "(" in quantum_prog_func:
                try:
                    func_name = quantum_prog_func.split("def ", 1)[1].split("(")[0].strip()
                except:
                    func_name = "pqctest"  # fallback
            else:
                func_name = quantum_prog_func or "pqctest"

            generated_code = f"""
            import pyqpanda3.core as pq
            import numpy as np
            from pyvqnet.qnn.pq3 import grad, ProbsMeasure

            {quantum_prog_func}

            # 计算梯度
            gradient = grad({func_name}, {input_params})
            print("梯度结果:", gradient)
            """

            return {
                "status": "success",
                "message": f"已生成量子梯度计算代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet grad API生成的真实代码。input_params是要计算梯度的参数列表。"
            }
        elif "AmplitudeEmbeddingCircuit" in tool_name:
            # AmplitudeEmbeddingCircuit 函数生成
            input_feat = arguments.get('input_feat')
            qubits = arguments.get('qubits')

            generated_code = f"""
            import numpy as np
            import pyqpanda3.core as pq
            from pyvqnet.qnn.pq3.template import AmplitudeEmbeddingCircuit

            # 输入特征
            input_feat = {input_feat}
            qlist = {qubits}

            # 创建量子虚拟机和线路
            machine = pq.CPUQVM()
            m_prog = pq.QProg()

            # 添加振幅编码线路
            cir = AmplitudeEmbeddingCircuit(input_feat, qlist)
            m_prog << cir
            """

            return {
                "status": "success",
                "message": f"已生成振幅编码线路代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet AmplitudeEmbeddingCircuit API生成的真实代码。input_feat需要是长度为2^n的数组，n为量子比特数。"
            }
        elif "AngleEmbeddingCircuit" in tool_name:
            # AngleEmbeddingCircuit 函数生成
            input_feat = arguments.get('input_feat')
            qubits = arguments.get('qubits')
            rotation = arguments.get('rotation')

            # 构建参数
            params = [input_feat, str(qubits)]
            if rotation is not None:
                rotation_str = f'"{rotation}"' if isinstance(rotation, str) else rotation
                params.append(f'rotation={rotation_str}')

            params_str = ", ".join(params)

            generated_code = f"""
            import numpy as np
            import pyqpanda3.core as pq
            from pyvqnet.qnn.pq3.template import AngleEmbeddingCircuit

            # 输入特征
            input_feat = {input_feat}
            qlist = {qubits}

            # 创建量子虚拟机和线路
            machine = pq.CPUQVM()
            m_prog = pq.QProg()

            # 添加角度编码线路
            cir = AngleEmbeddingCircuit({params_str})
            m_prog << cir
            """

            return {
                "status": "success",
                "message": f"已生成角度编码线路代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet AngleEmbeddingCircuit API生成的真实代码。rotation默认值为'X'，可选值为'X', 'Y', 'Z'。"
            }
        elif "pyvqnet.qnn.vqc.QMachine" in tool_name:
            # QMachine 类生成
            num_wires = arguments.get('num_wires')

            generated_code = f"""
            from pyvqnet.qnn.vqc import QMachine
            import pyvqnet

            # 创建量子虚拟机
            qm = QMachine({num_wires})

            # 打印初始状态
            print("初始量子状态:", qm.states)
            """

            return {
                "status": "success",
                "message": f"已生成QMachine创建代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet QMachine API生成的真实代码。num_wires是量子比特数量，默认使用CPU设备和kcomplex64数据类型。"
            }
        elif "pyvqnet.qnn.vqc.hadamard" in tool_name:
            # hadamard 门函数生成
            q_machine = arguments.get('q_machine')
            wires = arguments.get('wires')

            generated_code = f"""
            from pyvqnet.qnn.vqc import hadamard, QMachine

            # 应用Hadamard门
            hadamard(q_machine={q_machine}, wires={wires})

            # 打印应用后的状态
            print("应用Hadamard门后的状态:", {q_machine}.states)
            """

            return {
                "status": "success",
                "message": f"已生成Hadamard门应用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet hadamard API生成的真实代码。wires可以是单个比特索引或列表，可选参数params、use_dagger使用默认值。"
            }
        elif "pyvqnet.qnn.vqc.Probability" in tool_name:
            # Probability 测量类生成
            wires = arguments.get('wires')

            generated_code = f"""
            from pyvqnet.qnn.vqc import Probability, QMachine, hadamard

            # 创建概率测量实例
            prob_measure = Probability(wires={wires})

            # 执行测量（假设qm是已创建的QMachine实例）
            # probabilities = prob_measure(qm)
            # print("测量概率:", probabilities)
            """

            return {
                "status": "success",
                "message": f"已生成概率测量代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet Probability API生成的真实代码。wires是要测量的量子比特索引，可选参数name使用默认值。"
            }
        else:
            return {
                "status": "success",
                "message": f"模拟调用: {tool_name}",
                "parameters": arguments,
                "note": "这是模拟结果。请实现真实 API 调用。"
            }

    def run(self):
        """运行 stdio 服务器"""
        print("VQNET MCP 服务器启动...", file=sys.stderr)
        print(f"工作目录: {os.getcwd()}", file=sys.stderr)
        print(f"工具文件: {self.tools_file}", file=sys.stderr)

        # 设置无缓冲的 stdin/stdout
        sys.stdin.reconfigure(encoding='utf-8', errors='ignore')
        sys.stdout.reconfigure(encoding='utf-8', errors='ignore')

        while True:
            try:
                # 读取一行输入
                line = sys.stdin.readline()
                if not line:
                    print("输入结束，退出", file=sys.stderr)
                    break

                line = line.strip()
                if not line:
                    continue

                # 解析 JSON-RPC 请求
                try:
                    request = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"JSON 解析错误: {e}", file=sys.stderr)
                    print(f"输入行: {line[:100]}...", file=sys.stderr)
                    continue

                # 处理请求
                response = self.handle_request(request)

                # 发送响应
                if response.get("id") is not None:  # 只响应有 ID 的请求
                    response_json = json.dumps(response, ensure_ascii=False)
                    sys.stdout.write(response_json + "\n")
                    sys.stdout.flush()

            except KeyboardInterrupt:
                print("收到中断信号，退出", file=sys.stderr)
                break
            except Exception as e:
                print(f"服务器错误: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                # 继续运行，不退出

        print("服务器关闭", file=sys.stderr)

def main():
    """主函数"""
    # 检查工具文件
    tools_file = "vqnet_all_tools.json"
    if not os.path.exists(tools_file):
        print(f"错误: 工具文件 {tools_file} 不存在", file=sys.stderr)
        print("请先运行: python rst_to_mcp.py", file=sys.stderr)
        sys.exit(1)

    # 创建并运行服务器
    server = SimpleMCPServer(tools_file)
    server.run()

if __name__ == "__main__":
    main()