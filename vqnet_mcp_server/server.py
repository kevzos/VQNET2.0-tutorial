#!/usr/bin/env python
"""
简单的 MCP stdio 服务器实现
专为 Claude Code 集成设计

实现基本的 MCP 协议，通过 stdio 与 Claude Code 通信。
"""

import json
import sys
import traceback
import importlib
import os
from typing import Dict, Any, List, Optional

class SimpleMCPServer:
    """简单的 MCP 服务器实现"""

    def __init__(self, tools_file: str = None):
        # 如果未指定，使用包内默认的工具文件
        self.tools_file = tools_file
        self.tools = self.load_tools()

    def load_tools(self) -> List[Dict[str, Any]]:
        """从文件加载工具定义，支持包内数据文件"""
        # 如果指定了外部文件，优先使用外部文件
        if self.tools_file is not None and os.path.exists(self.tools_file):
            try:
                with open(self.tools_file, 'r', encoding='utf-8') as f:
                    tools = json.load(f)
                print(f"已加载 {len(tools)} 个工具定义 (外部文件)", file=sys.stderr)
                return tools
            except FileNotFoundError:
                print(f"错误: 工具文件 {self.tools_file} 未找到", file=sys.stderr)
            except json.JSONDecodeError as e:
                print(f"错误: 无法解析 JSON 文件: {e}", file=sys.stderr)

        # 尝试从包内加载
        try:
            # Python 3.9+ 推荐写法
            from importlib.resources import files
            tools = json.load(files("vqnet_mcp_server").joinpath("vqnet_all_tools.json").open('r', encoding='utf-8'))
            print(f"已加载 {len(tools)} 个工具定义 (包内)", file=sys.stderr)
            return tools
        except (ImportError, AttributeError):
            # Python 3.8 兼容写法
            try:
                tools = json.load(importlib.resources.open_text("vqnet_mcp_server", "vqnet_all_tools.json"))
                print(f"已加载 {len(tools)} 个工具定义 (包内兼容模式)", file=sys.stderr)
                return tools
            except Exception as e:
                print(f"错误: 无法从包内加载工具文件: {e}", file=sys.stderr)
                print("请确保包已正确安装，或通过参数指定工具文件路径", file=sys.stderr)
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
                    "version": "0.1.0"
                }
            }
        }

    def handle_tools_list(self, request: Dict[str, Any], request_id: Any) -> Dict[str, Any]:
        """处理工具列表请求"""
        # 转换工具格式为 MCP 格式
        mcp_tools = []
        for tool in self.tools:
            mcp_tool = {
                "name": tool.get("name", ""),
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
        name = params.get("name", "")
        arguments = params.get("arguments", {})

        # 查找工具定义
        tool_def = None
        for t in self.tools:
            if t.get("name") == name:
                tool_def = t
                break

        if tool_def is None:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": f"工具不存在: {name}"
                }
            }

        try:
            # 这里是实际执行工具的地方
            # 根据 VQNET API 执行调用并返回结果
            # 当前版本仅返回工具信息，实际执行需要导入 VQNET
            result = {
                "content": [
                    {
                        "type": "text",
                        "text": f"调用工具: {name}\n参数: {json.dumps(arguments, indent=2, ensure_ascii=False)}\n\n注意: 需要安装 VQNET 后才能实际执行API调用。"
                    }
                ]
            }
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
        except Exception as e:
            print(f"工具执行出错: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"工具执行错误: {str(e)}"
                }
            }

    def run(self):
        """运行服务器，通过 stdio 处理请求"""
        print("VQNET MCP 服务器启动", file=sys.stderr)
        print(f"工具数量: {len(self.tools)}", file=sys.stderr)

        for line in sys.stdin:
            try:
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
    """主函数 - 入口点"""
    # 检查是否通过命令行参数指定了工具文件
    tools_file = None
    if len(sys.argv) > 1:
        tools_file = sys.argv[1]

    # 创建并运行服务器
    server = SimpleMCPServer(tools_file)
    server.run()

if __name__ == "__main__":
    main()
