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

    def __init__(self, tools_file: str = "qnn_pq3_mcp_tools.json"):
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
    tools_file = "qnn_pq3_mcp_tools.json"
    if not os.path.exists(tools_file):
        print(f"错误: 工具文件 {tools_file} 不存在", file=sys.stderr)
        print("请先运行: python rst_to_mcp.py", file=sys.stderr)
        sys.exit(1)

    # 创建并运行服务器
    server = SimpleMCPServer(tools_file)
    server.run()

if __name__ == "__main__":
    main()