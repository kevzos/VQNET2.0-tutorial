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
            params = []
            if data is not None:
                params.append(str(data))
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
        elif "pyvqnet.tensor.ones" in tool_name or "ones" in tool_name:
            # ones 函数生成
            shape = arguments.get('shape')
            device = arguments.get('device', 'pyvqnet.DEV_CPU')
            dtype = arguments.get('dtype', None)

            # 格式化参数
            shape_str = str(shape) if isinstance(shape, (list, tuple)) else shape
            dtype_str = dtype if dtype is not None else 'None'

            generated_code = f"""
            from pyvqnet.tensor import ones
            from pyvqnet.dtype import *

            # 生成全1张量
            tensor = ones(
                shape={shape_str},
                device={device},
                dtype={dtype_str}
            )
            print("全1张量形状:", tensor.shape)
            print("全1张量:", tensor)
            """

            return {
                "status": "success",
                "message": f"已生成ones函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet ones API生成的真实代码，生成指定形状的全1张量。"
            }
        elif "pyvqnet.tensor.zeros" in tool_name or "zeros" in tool_name:
            # zeros 函数生成
            shape = arguments.get('shape')
            device = arguments.get('device', 'pyvqnet.DEV_CPU')
            dtype = arguments.get('dtype', None)

            # 格式化参数
            shape_str = str(shape) if isinstance(shape, (list, tuple)) else shape
            dtype_str = dtype if dtype is not None else 'None'

            generated_code = f"""
            from pyvqnet.tensor import zeros
            from pyvqnet.dtype import *

            # 生成全0张量
            tensor = zeros(
                shape={shape_str},
                device={device},
                dtype={dtype_str}
            )
            print("全0张量形状:", tensor.shape)
            print("全0张量:", tensor)
            """

            return {
                "status": "success",
                "message": f"已生成zeros函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet zeros API生成的真实代码，生成指定形状的全0张量。"
            }
        elif "pyvqnet.tensor.arange" in tool_name or "arange" in tool_name:
            # arange 函数生成
            start = arguments.get('start')
            end = arguments.get('end')
            step = arguments.get('step', 1)
            device = arguments.get('device', 'pyvqnet.DEV_CPU')
            dtype = arguments.get('dtype', None)
            requires_grad = arguments.get('requires_grad', False)

            # 格式化参数
            requires_grad_str = str(requires_grad) if isinstance(requires_grad, bool) else requires_grad
            dtype_str = dtype if dtype is not None else 'None'

            generated_code = f"""
            from pyvqnet.tensor import arange
            from pyvqnet.dtype import *

            # 生成序列张量
            tensor = arange(
                start={start},
                end={end},
                step={step},
                device={device},
                dtype={dtype_str},
                requires_grad={requires_grad_str}
            )
            print("序列张量形状:", tensor.shape)
            print("序列张量:", tensor)
            """

            return {
                "status": "success",
                "message": f"已生成arange函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet arange API生成的真实代码，生成[start, end)范围内步长为step的序列张量。"
            }
        elif "pyvqnet.tensor.randn" in tool_name or "randn" in tool_name:
            # randn 函数生成
            shape = arguments.get('shape')
            mean = arguments.get('mean', 0.0)
            std = arguments.get('std', 1.0)
            device = arguments.get('device', 'pyvqnet.DEV_CPU')
            dtype = arguments.get('dtype', None)
            requires_grad = arguments.get('requires_grad', False)

            # 格式化参数
            shape_str = str(shape) if isinstance(shape, (list, tuple)) else shape
            requires_grad_str = str(requires_grad) if isinstance(requires_grad, bool) else requires_grad
            dtype_str = dtype if dtype is not None else 'None'

            generated_code = f"""
            from pyvqnet.tensor import randn
            from pyvqnet.dtype import *

            # 生成正态分布随机张量
            tensor = randn(
                shape={shape_str},
                mean={mean},
                std={std},
                device={device},
                dtype={dtype_str},
                requires_grad={requires_grad_str}
            )
            print("随机张量形状:", tensor.shape)
            print("随机张量:", tensor)
            """

            return {
                "status": "success",
                "message": f"已生成randn函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet randn API生成的真实代码，生成指定均值和标准差的正态分布随机张量。"
            }
        elif "pyvqnet.tensor.add" in tool_name or "add" in tool_name:
            # add 函数生成
            t1 = arguments.get('t1')
            t2 = arguments.get('t2')

            generated_code = f"""
            from pyvqnet.tensor import add, QTensor
            import numpy as np

            # 创建输入张量
            t1 = {t1}
            t2 = {t2}

            # 张量加法
            result = add(t1, t2)
            print("加法结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成add函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet add API生成的真实代码，实现两个张量的逐元素加法。"
            }
        elif "pyvqnet.tensor.mul" in tool_name or "mul" in tool_name:
            # mul 函数生成
            t1 = arguments.get('t1')
            t2 = arguments.get('t2')

            generated_code = f"""
            from pyvqnet.tensor import mul, QTensor
            import numpy as np

            # 创建输入张量
            t1 = {t1}
            t2 = {t2}

            # 张量乘法
            result = mul(t1, t2)
            print("乘法结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成mul函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet mul API生成的真实代码，实现两个张量的逐元素乘法。"
            }
        elif "pyvqnet.tensor.matmul" in tool_name or "matmul" in tool_name:
            # matmul 函数生成
            t1 = arguments.get('t1')
            t2 = arguments.get('t2')

            generated_code = f"""
            from pyvqnet.tensor import matmul, QTensor
            import numpy as np

            # 创建输入张量
            t1 = {t1}
            t2 = {t2}

            # 矩阵乘法
            result = matmul(t1, t2)
            print("矩阵乘法结果形状:", result.shape)
            print("矩阵乘法结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成matmul函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet matmul API生成的真实代码，实现两个张量的矩阵乘法。"
            }
        elif "pyvqnet.tensor.concatenate" in tool_name or "concatenate" in tool_name:
            # concatenate 函数生成
            args = arguments.get('args')
            axis = arguments.get('axis', 1)

            # 格式化参数
            args_str = str(args) if isinstance(args, list) else args

            generated_code = f"""
            from pyvqnet.tensor import concatenate, QTensor
            import numpy as np

            # 创建输入张量列表
            tensors = {args_str}

            # 张量拼接
            result = concatenate(tensors, axis={axis})
            print("拼接后张量形状:", result.shape)
            print("拼接后张量:", result)
            """

            return {
                "status": "success",
                "message": f"已生成concatenate函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet concatenate API生成的真实代码，在指定轴上拼接多个张量。"
            }
        elif "pyvqnet.tensor.flatten" in tool_name or "flatten" in tool_name:
            # flatten 函数生成
            t = arguments.get('t')
            start = arguments.get('start', 0)
            end = arguments.get('end', -1)

            generated_code = f"""
            from pyvqnet.tensor import flatten, QTensor
            import numpy as np

            # 创建输入张量
            tensor = {t}

            # 张量展平
            result = flatten(tensor, start={start}, end={end})
            print("展平后张量形状:", result.shape)
            print("展平后张量:", result)
            """

            return {
                "status": "success",
                "message": f"已生成flatten函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet flatten API生成的真实代码，将张量从start到end的维度展平为一维。"
            }
        elif "pyvqnet.tensor.reshape" in tool_name or "reshape" in tool_name:
            # reshape 函数生成
            t = arguments.get('t')
            new_shape = arguments.get('new_shape')

            # 格式化参数
            new_shape_str = str(new_shape) if isinstance(new_shape, (list, tuple)) else new_shape

            generated_code = f"""
            from pyvqnet.tensor import reshape, QTensor
            import numpy as np

            # 创建输入张量
            tensor = {t}

            # 张量重塑
            result = reshape(tensor, new_shape={new_shape_str})
            print("重塑后张量形状:", result.shape)
            print("重塑后张量:", result)
            """

            return {
                "status": "success",
                "message": f"已生成reshape函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet reshape API生成的真实代码，将张量重塑为指定形状。"
            }
        elif "pyvqnet.tensor.full" in tool_name or "full" in tool_name:
            # full 函数生成
            shape = arguments.get('shape')
            value = arguments.get('value')
            device = arguments.get('device', 'pyvqnet.DEV_CPU')
            dtype = arguments.get('dtype', None)

            # 格式化参数
            shape_str = str(shape) if isinstance(shape, (list, tuple)) else shape
            dtype_str = dtype if dtype is not None else 'None'

            generated_code = f"""
            from pyvqnet.tensor import full
            from pyvqnet.dtype import *

            # 生成指定值的张量
            tensor = full(
                shape={shape_str},
                value={value},
                device={device},
                dtype={dtype_str}
            )
            print("指定值张量形状:", tensor.shape)
            print("指定值张量:", tensor)
            """

            return {
                "status": "success",
                "message": f"已生成full函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet full API生成的真实代码，生成指定形状并用指定值填充的张量。"
            }
        elif "pyvqnet.tensor.eye" in tool_name or "eye" in tool_name:
            # eye 函数生成
            size = arguments.get('size')
            offset = arguments.get('offset', 0)
            device = arguments.get('device', 'pyvqnet.DEV_CPU')
            dtype = arguments.get('dtype', None)

            # 格式化参数
            dtype_str = dtype if dtype is not None else 'None'

            generated_code = f"""
            from pyvqnet.tensor import eye
            from pyvqnet.dtype import *

            # 生成单位矩阵
            tensor = eye(
                size={size},
                offset={offset},
                device={device},
                dtype={dtype_str}
            )
            print("单位矩阵形状:", tensor.shape)
            print("单位矩阵:", tensor)
            """

            return {
                "status": "success",
                "message": f"已生成eye函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet eye API生成的真实代码，生成指定大小的单位矩阵。"
            }
        elif "pyvqnet.tensor.exp" in tool_name or "exp" in tool_name:
            # exp 函数生成
            t = arguments.get('t')

            generated_code = f"""
            from pyvqnet.tensor import exp, QTensor
            import numpy as np

            # 创建输入张量
            tensor = {t}

            # 指数计算
            result = exp(tensor)
            print("指数计算结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成exp函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet exp API生成的真实代码，计算张量的自然指数。"
            }
        elif "pyvqnet.tensor.log" in tool_name or "log" in tool_name:
            # log 函数生成
            t = arguments.get('t')

            generated_code = f"""
            from pyvqnet.tensor import log, QTensor
            import numpy as np

            # 创建输入张量
            tensor = {t}

            # 对数计算
            result = log(tensor)
            print("对数计算结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成log函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet log API生成的真实代码，计算张量的自然对数。"
            }
        elif "pyvqnet.tensor.power" in tool_name or "power" in tool_name:
            # power 函数生成
            t1 = arguments.get('t1')
            t2 = arguments.get('t2')

            generated_code = f"""
            from pyvqnet.tensor import power, QTensor
            import numpy as np

            # 创建输入张量
            base = {t1}
            exponent = {t2}

            # 幂运算
            result = power(base, exponent)
            print("幂运算结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成power函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet power API生成的真实代码，计算张量的幂运算。"
            }
        elif "pyvqnet.tensor.sums" in tool_name or "sums" in tool_name:
            # sums 函数生成
            t = arguments.get('t')
            axis = arguments.get('axis', None)
            keepdims = arguments.get('keepdims', False)

            # 格式化参数
            axis_str = str(axis) if axis is not None else 'None'
            keepdims_str = str(keepdims) if isinstance(keepdims, bool) else keepdims

            generated_code = f"""
            from pyvqnet.tensor import sums, QTensor
            import numpy as np

            # 创建输入张量
            tensor = {t}

            # 求和运算
            result = sums(tensor, axis={axis_str}, keepdims={keepdims_str})
            print("求和结果形状:", result.shape)
            print("求和结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成sums函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet sums API生成的真实代码，在指定轴上对张量进行求和。"
            }
        elif "pyvqnet.tensor.mean" in tool_name or "mean" in tool_name:
            # mean 函数生成
            t = arguments.get('t')
            axis = arguments.get('axis', None)
            keepdims = arguments.get('keepdims', False)

            # 格式化参数
            axis_str = str(axis) if axis is not None else 'None'
            keepdims_str = str(keepdims) if isinstance(keepdims, bool) else keepdims

            generated_code = f"""
            from pyvqnet.tensor import mean, QTensor
            import numpy as np

            # 创建输入张量
            tensor = {t}

            # 求平均值
            result = mean(tensor, axis={axis_str}, keepdims={keepdims_str})
            print("平均值结果形状:", result.shape)
            print("平均值结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成mean函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet mean API生成的真实代码，在指定轴上计算张量的平均值。"
            }
        elif "pyvqnet.tensor.transpose" in tool_name or "transpose" in tool_name:
            # transpose 函数生成
            t = arguments.get('t')
            dim = arguments.get('dim')

            # 格式化参数
            dim_str = str(dim) if isinstance(dim, list) else dim

            generated_code = f"""
            from pyvqnet.tensor import transpose, QTensor
            import numpy as np

            # 创建输入张量
            tensor = {t}

            # 矩阵转置
            result = transpose(tensor, dim={dim_str})
            print("转置后形状:", result.shape)
            print("转置结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成transpose函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet transpose API生成的真实代码，按照指定维度顺序转置张量。"
            }
        elif "pyvqnet.tensor.squeeze" in tool_name or "squeeze" in tool_name:
            # squeeze 函数生成
            t = arguments.get('t')
            axis = arguments.get('axis', -1)

            generated_code = f"""
            from pyvqnet.tensor import squeeze, QTensor
            import numpy as np

            # 创建输入张量
            tensor = {t}

            # 去除单维度
            result = squeeze(tensor, axis={axis})
            print("squeeze后形状:", result.shape)
            print("squeeze结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成squeeze函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet squeeze API生成的真实代码，去除张量中维度大小为1的轴。"
            }
        elif "pyvqnet.tensor.greater" in tool_name or "greater" in tool_name:
            # greater 函数生成
            t1 = arguments.get('t1')
            t2 = arguments.get('t2')

            generated_code = f"""
            from pyvqnet.tensor import greater, QTensor
            import numpy as np

            # 创建输入张量
            t1 = {t1}
            t2 = {t2}

            # 大于比较
            result = greater(t1, t2)
            print("大于比较结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成greater函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet greater API生成的真实代码，逐元素比较两个张量是否满足t1 > t2。"
            }
        elif "pyvqnet.qnn.pq3.quantumlayer.QpandaQProgVQCLayer" in tool_name or "QuantumLayerV3" in tool_name:
            # QpandaQProgVQCLayer 类生成（别名QuantumLayerV3）
            origin_qprog_func = arguments.get('origin_qprog_func')
            para_num = arguments.get('para_num')
            qvm_type = arguments.get('qvm_type', 'cpu')
            pauli_str_dict = arguments.get('pauli_str_dict', None)
            shots = arguments.get('shots', 1000)
            initializer = arguments.get('initializer', None)
            dtype = arguments.get('dtype', None)
            name = arguments.get('name', "")

            if origin_qprog_func.startswith("def ") and "(" in origin_qprog_func:
                try:
                    func_name = origin_qprog_func.split("def ", 1)[1].split("(")[0].strip()
                except:
                    func_name = "qfun"  # fallback
            else:
                func_name = origin_qprog_func or "qfun"

            # 格式化参数
            qvm_type_str = f'"{qvm_type}"' if isinstance(qvm_type, str) else qvm_type
            pauli_str_dict_str = pauli_str_dict if pauli_str_dict is not None else 'None'
            initializer_str = initializer if initializer is not None else 'None'
            dtype_str = dtype if dtype is not None else 'None'
            name_str = f'"{name}"' if isinstance(name, str) else name

            generated_code = f"""
            import pyqpanda3.core as pq
            import pyvqnet
            from pyvqnet.qnn.pq3.quantumlayer import QpandaQProgVQCLayer

            {origin_qprog_func}

            # 创建QpandaQProgVQCLayer实例（别名QuantumLayerV3）
            quantum_layer = QpandaQProgVQCLayer(
                {func_name},
                {para_num},
                qvm_type={qvm_type_str},
                pauli_str_dict={pauli_str_dict_str},
                shots={shots},
                initializer={initializer_str},
                dtype={dtype_str},
                name={name_str}
            )
            """

            return {
                "status": "success",
                "message": f"已生成QpandaQProgVQCLayer（QuantumLayerV3）创建代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet QpandaQProgVQCLayer API生成的真实代码。确保origin_qprog_func返回pyqpanda3.core.QProg类型。"
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
        elif "pyvqnet.qnn.qlinear.QLinear" in tool_name or "QLinear" in tool_name:
            # QLinear 量子全连接层
            input_channels = arguments.get('input_channels')
            output_channels = arguments.get('output_channels')
            machine = arguments.get('machine', '"CPU"')

            generated_code = f"""
            from pyvqnet.tensor import QTensor
            from pyvqnet.qnn.qlinear import QLinear

            # 创建量子全连接层
            m = QLinear({input_channels}, {output_channels}, machine={machine})

            # 示例输入
            params = [[0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452],
            [1.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452],
            [1.37454012, 1.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452],
            [1.37454012, 1.95071431, 1.73199394, 1.59865848, 0.15601864, 0.15599452]]
            input = QTensor(params, requires_grad=True)

            # 前向传播
            output = m(input)
            print("量子全连接层输出:", output)
            """

            return {
                "status": "success",
                "message": f"已生成QLinear量子全连接层代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "QLinear是量子全连接层，不带变分量子参数，支持CPU/GPU模拟。"
            }
        elif "pyvqnet.qnn.qcnn.qconv.QConv" in tool_name or "QConv" in tool_name:
            # QConv 量子卷积层
            input_channels = arguments.get('input_channels')
            output_channels = arguments.get('output_channels')
            quantum_number = arguments.get('quantum_number')
            stride = arguments.get('stride', '(1, 1)')
            padding = arguments.get('padding', '(0, 0)')
            machine = arguments.get('machine', '"CPU"')

            generated_code = f"""
            from pyvqnet.tensor import tensor
            from pyvqnet.qnn.qcnn.qconv import QConv

            # 创建量子卷积层
            layer = QConv(
                input_channels={input_channels},
                output_channels={output_channels},
                quantum_number={quantum_number},
                stride={stride},
                padding={padding},
                machine={machine}
            )

            # 示例输入
            x = tensor.ones([1, {input_channels}, 4, 4])

            # 前向传播
            y = layer(x)
            print("量子卷积层输出形状:", y.shape)
            print("量子卷积层输出:", y)
            """

            return {
                "status": "success",
                "message": f"已生成QConv量子卷积层代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "QConv是量子卷积层，用量子线路替代传统卷积核，支持自定义卷积核大小和步长。"
            }
        elif "pyvqnet.qnn.pq3.template.BasicEmbeddingCircuit" in tool_name or "BasicEmbeddingCircuit" in tool_name:
            # BasicEmbeddingCircuit 基态编码线路
            input_feat = arguments.get('input_feat', 'tensor.QTensor([1,1,0])')
            qlist = arguments.get('qlist', 'range(3)')

            generated_code = f"""
            from pyvqnet.qnn.pq3.template import BasicEmbeddingCircuit
            import pyqpanda3.core as pq
            from pyvqnet import tensor

            # 输入特征
            input_feat = {input_feat}
            qlist = {qlist}

            # 创建基态编码线路
            circuit = BasicEmbeddingCircuit(input_feat, qlist)
            print("基态编码线路:", circuit)
            """

            return {
                "status": "success",
                "message": f"已生成BasicEmbeddingCircuit基态编码代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "BasicEmbeddingCircuit将二进制特征编码为量子比特的基态，如[1,1,0]对应|110>态。"
            }
        elif "pyvqnet.qnn.pq3.template.IQPEmbeddingCircuits" in tool_name or "IQPEmbeddingCircuits" in tool_name:
            # IQPEmbeddingCircuits IQP编码线路
            input_feat = arguments.get('input_feat', 'np.arange(1,100)')
            qubits = arguments.get('qubits', 'range(3)')
            rep = arguments.get('rep', 1)

            generated_code = f"""
            import numpy as np
            from pyvqnet.qnn.pq3.template import IQPEmbeddingCircuits

            # 输入特征
            input_feat = {input_feat}
            qlist = {qubits}

            # 创建IQP编码线路
            circuit = IQPEmbeddingCircuits(input_feat, qlist, rep={rep})
            print("IQP编码线路长度:", len(circuit))
            """

            return {
                "status": "success",
                "message": f"已生成IQPEmbeddingCircuits IQP编码代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "IQPEmbeddingCircuits使用IQP线路的对角门编码特征，支持重复线路块增强表达能力。"
            }
        elif "pyvqnet.qnn.pq3.template.RotCircuit" in tool_name or "RotCircuit" in tool_name:
            # RotCircuit 单量子比特任意旋转
            para = arguments.get('para', 'tensor.QTensor([3,4,5])')
            qubits = arguments.get('qubits', 1)

            generated_code = f"""
            from pyvqnet.qnn.pq3.template import RotCircuit
            import pyqpanda3.core as pq
            from pyvqnet import tensor

            # 旋转参数 [phi, theta, omega]
            param = {para}
            m_qlist = {qubits}

            # 创建单量子比特任意旋转线路
            c = RotCircuit(param, m_qlist)
            print("单量子比特旋转线路:", c)
            """

            return {
                "status": "success",
                "message": f"已生成RotCircuit单量子比特旋转代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "RotCircuit实现任意单量子比特旋转，等价于RZ(omega)RY(theta)RZ(phi)组合。"
            }
        elif "pyvqnet.qnn.pq3.template.CRotCircuit" in tool_name or "CRotCircuit" in tool_name:
            # CRotCircuit 受控Rot操作
            para = arguments.get('para', '[0.5, 1.0, 1.5]')
            control_qubits = arguments.get('control_qubits', 0)
            rot_qubits = arguments.get('rot_qubits', 1)

            generated_code = f"""
            from pyvqnet.qnn.pq3.template import CRotCircuit
            import pyqpanda3.core as pq

            # 旋转参数 [phi, theta, omega]
            param = {para}
            control_qubit = {control_qubits}
            target_qubit = {rot_qubits}

            # 创建受控Rot线路
            c = CRotCircuit(param, control_qubit, target_qubit)
            print("受控Rot线路:", c)
            """

            return {
                "status": "success",
                "message": f"已生成CRotCircuit受控旋转代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "CRotCircuit实现受控的任意单量子比特旋转操作，control_qubit为控制位，rot_qubits为目标位。"
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
        elif "pyvqnet.qnn.vqc.MeasureAll" in tool_name or "MeasureAll" in tool_name:
            # MeasureAll 全量子比特测量生成
            obs = arguments.get('obs')
            name = arguments.get('name', "")

            name_str = f'"{name}"' if isinstance(name, str) else name

            generated_code = f"""
            from pyvqnet.qnn.vqc import MeasureAll, QMachine, hadamard
            from pyvqnet.tensor import QTensor
            import numpy as np

            # 创建量子虚拟机
            num_qubits = 2
            qm = QMachine(num_qubits)
            qm.reset_states(num_qubits)

            # 制备量子态
            hadamard(q_machine=qm, wires=0)
            hadamard(q_machine=qm, wires=1)

            # 定义可观测量
            obs = np.array([[1, 0], [0, -1]], dtype=np.complex64)  # 泡利Z矩阵

            # 创建全量子比特测量
            measure = MeasureAll(
                obs={obs if obs is not None else 'obs'},
                name={name_str}
            )

            # 执行测量
            result = measure(q_machine=qm)
            print("全测量结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成MeasureAll全量子比特测量代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet MeasureAll API生成的真实代码，对所有量子比特进行测量。"
            }
        elif "pyvqnet.qnn.vqc.cnot" in tool_name or "cnot" in tool_name:
            # cnot 受控非门生成
            q_machine = arguments.get('q_machine', 'qm')
            wires = arguments.get('wires')
            params = arguments.get('params', None)
            use_dagger = arguments.get('use_dagger', False)

            # 格式化参数
            wires_str = str(wires) if isinstance(wires, (list, tuple)) else wires
            params_str = params if params is not None else 'None'
            use_dagger_str = str(use_dagger) if isinstance(use_dagger, bool) else use_dagger

            generated_code = f"""
            from pyvqnet.qnn.vqc import QMachine, cnot, hadamard, measure_all
            import numpy as np

            # 创建量子虚拟机
            qm = QMachine(2)
            qm.reset_states(2)

            # 制备量子态
            hadamard(q_machine=qm, wires=0)

            # 应用CNOT门
            cnot(
                q_machine=qm,
                wires={wires_str if wires is not None else '[0, 1]'},
                params={params_str},
                use_dagger={use_dagger_str}
            )

            # 测量
            result = measure_all(q_machine=qm, shots=1000)
            print("CNOT门测量结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成cnot受控非门代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet cnot API生成的真实代码，实现受控非门操作。wires参数格式为[控制比特, 目标比特]。"
            }
        elif "pyvqnet.qnn.vqc.swap" in tool_name or "swap" in tool_name:
            # swap 交换门生成
            q_machine = arguments.get('q_machine', 'qm')
            wires = arguments.get('wires')
            params = arguments.get('params', None)
            use_dagger = arguments.get('use_dagger', False)

            # 格式化参数
            wires_str = str(wires) if isinstance(wires, (list, tuple)) else wires
            params_str = params if params is not None else 'None'
            use_dagger_str = str(use_dagger) if isinstance(use_dagger, bool) else use_dagger

            generated_code = f"""
            from pyvqnet.qnn.vqc import QMachine, swap, pauliX, measure_all
            import numpy as np

            # 创建量子虚拟机
            qm = QMachine(2)
            qm.reset_states(2)

            # 制备量子态：q0=|1>, q1=|0>
            pauliX(q_machine=qm, wires=0)

            # 应用SWAP门
            swap(
                q_machine=qm,
                wires={wires_str if wires is not None else '[0, 1]'},
                params={params_str},
                use_dagger={use_dagger_str}
            )

            # 测量
            result = measure_all(q_machine=qm, shots=1000)
            print("SWAP门测量结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成swap交换门代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet swap API生成的真实代码，实现两个量子比特的状态交换。"
            }
        elif "pyvqnet.qnn.vqc.toffoli" in tool_name or "toffoli" in tool_name:
            # toffoli 托佛利门生成
            q_machine = arguments.get('q_machine', 'qm')
            wires = arguments.get('wires')
            params = arguments.get('params', None)
            use_dagger = arguments.get('use_dagger', False)

            # 格式化参数
            wires_str = str(wires) if isinstance(wires, (list, tuple)) else wires
            params_str = params if params is not None else 'None'
            use_dagger_str = str(use_dagger) if isinstance(use_dagger, bool) else use_dagger

            generated_code = f"""
            from pyvqnet.qnn.vqc import QMachine, toffoli, pauliX, measure_all
            import numpy as np

            # 创建量子虚拟机
            qm = QMachine(3)
            qm.reset_states(3)

            # 制备量子态：q0=|1>, q1=|1>, q2=|0>
            pauliX(q_machine=qm, wires=0)
            pauliX(q_machine=qm, wires=1)

            # 应用Toffoli门（控制位0,1，目标位2）
            toffoli(
                q_machine=qm,
                wires={wires_str if wires is not None else '[0, 1, 2]'},
                params={params_str},
                use_dagger={use_dagger_str}
            )

            # 测量
            result = measure_all(q_machine=qm, shots=1000)
            print("Toffoli门测量结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成toffoli托佛利门代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet toffoli API生成的真实代码，实现三量子比特的Toffoli（受控-受控-非）门操作。"
            }
        elif "pyvqnet.qnn.vqc.cz" in tool_name or "cz" in tool_name:
            # cz 受控Z门生成
            q_machine = arguments.get('q_machine', 'qm')
            wires = arguments.get('wires')
            params = arguments.get('params', None)
            use_dagger = arguments.get('use_dagger', False)

            # 格式化参数
            wires_str = str(wires) if isinstance(wires, (list, tuple)) else wires
            params_str = params if params is not None else 'None'
            use_dagger_str = str(use_dagger) if isinstance(use_dagger, bool) else use_dagger

            generated_code = f"""
            from pyvqnet.qnn.vqc import QMachine, cz, hadamard, measure_all
            import numpy as np

            # 创建量子虚拟机
            qm = QMachine(2)
            qm.reset_states(2)

            # 制备量子态
            hadamard(q_machine=qm, wires=0)
            hadamard(q_machine=qm, wires=1)

            # 应用CZ门
            cz(
                q_machine=qm,
                wires={wires_str if wires is not None else '[0, 1]'},
                params={params_str},
                use_dagger={use_dagger_str}
            )

            # 测量
            result = measure_all(q_machine=qm, shots=1000)
            print("CZ门测量结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成cz受控Z门代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet cz API生成的真实代码，实现受控Z门操作。"
            }
        elif "pyvqnet.qnn.vqc.rx" in tool_name or "rx" in tool_name:
            # rx 绕X轴旋转门生成
            q_machine = arguments.get('q_machine', 'qm')
            wires = arguments.get('wires')
            params = arguments.get('params')
            use_dagger = arguments.get('use_dagger', False)

            # 格式化参数
            params_str = str(params) if params is not None else 'np.pi/2'
            use_dagger_str = str(use_dagger) if isinstance(use_dagger, bool) else use_dagger

            generated_code = f"""
            from pyvqnet.qnn.vqc import QMachine, rx, measure_all
            import numpy as np

            # 创建量子虚拟机
            qm = QMachine(1)
            qm.reset_states(1)

            # 应用RX门（绕X轴旋转90度）
            rx(
                q_machine=qm,
                wires={wires if wires is not None else 0},
                params={params_str},
                use_dagger={use_dagger_str}
            )

            # 测量
            result = measure_all(q_machine=qm, shots=1000)
            print("RX门测量结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成rx绕X轴旋转门代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet rx API生成的真实代码，实现绕X轴的旋转门操作。params参数为旋转角度。"
            }
        elif "pyvqnet.qnn.vqc.ry" in tool_name or "ry" in tool_name:
            # ry 绕Y轴旋转门生成
            q_machine = arguments.get('q_machine', 'qm')
            wires = arguments.get('wires')
            params = arguments.get('params')
            use_dagger = arguments.get('use_dagger', False)

            # 格式化参数
            params_str = str(params) if params is not None else 'np.pi/2'
            use_dagger_str = str(use_dagger) if isinstance(use_dagger, bool) else use_dagger

            generated_code = f"""
            from pyvqnet.qnn.vqc import QMachine, ry, measure_all
            import numpy as np

            # 创建量子虚拟机
            qm = QMachine(1)
            qm.reset_states(1)

            # 应用RY门（绕Y轴旋转90度）
            ry(
                q_machine=qm,
                wires={wires if wires is not None else 0},
                params={params_str},
                use_dagger={use_dagger_str}
            )

            # 测量
            result = measure_all(q_machine=qm, shots=1000)
            print("RY门测量结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成ry绕Y轴旋转门代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet ry API生成的真实代码，实现绕Y轴的旋转门操作。params参数为旋转角度。"
            }
        elif "pyvqnet.qnn.vqc.rz" in tool_name or "rz" in tool_name:
            # rz 绕Z轴旋转门生成
            q_machine = arguments.get('q_machine', 'qm')
            wires = arguments.get('wires')
            params = arguments.get('params')
            use_dagger = arguments.get('use_dagger', False)

            # 格式化参数
            params_str = str(params) if params is not None else 'np.pi/2'
            use_dagger_str = str(use_dagger) if isinstance(use_dagger, bool) else use_dagger

            generated_code = f"""
            from pyvqnet.qnn.vqc import QMachine, rz, hadamard, measure_all
            import numpy as np

            # 创建量子虚拟机
            qm = QMachine(1)
            qm.reset_states(1)

            # 制备叠加态
            hadamard(q_machine=qm, wires=0)

            # 应用RZ门（绕Z轴旋转90度）
            rz(
                q_machine=qm,
                wires={wires if wires is not None else 0},
                params={params_str},
                use_dagger={use_dagger_str}
            )

            # 测量
            result = measure_all(q_machine=qm, shots=1000)
            print("RZ门测量结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成rz绕Z轴旋转门代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet rz API生成的真实代码，实现绕Z轴的旋转门操作。params参数为旋转角度。"
            }
        elif "pyvqnet.qnn.vqc.VQC_HardwareEfficientAnsatz" in tool_name or "VQC_HardwareEfficientAnsatz" in tool_name:
            # VQC_HardwareEfficientAnsatz 硬件高效ansatz生成
            n_qubits = arguments.get('n_qubits')
            single_rot_gate_list = arguments.get('single_rot_gate_list')
            entangle_gate = arguments.get('entangle_gate', '"CNOT"')
            entangle_rules = arguments.get('entangle_rules', '"linear"')
            depth = arguments.get('depth', 1)
            initial = arguments.get('initial', None)
            dtype = arguments.get('dtype', None)

            # 格式化参数
            single_rot_gate_list_str = str(single_rot_gate_list) if isinstance(single_rot_gate_list, list) else '["RY", "RZ"]'
            initial_str = initial if initial is not None else 'None'
            dtype_str = dtype if dtype is not None else 'None'

            generated_code = f"""
            from pyvqnet.qnn.vqc import VQC_HardwareEfficientAnsatz, QMachine
            import numpy as np

            # 创建硬件高效ansatz
            ansatz = VQC_HardwareEfficientAnsatz(
                n_qubits={n_qubits if n_qubits is not None else 3},
                single_rot_gate_list={single_rot_gate_list_str},
                entangle_gate={entangle_gate},
                entangle_rules={entangle_rules},
                depth={depth},
                initial={initial_str},
                dtype={dtype_str}
            )

            # 获取线路参数数量
            param_count = ansatz.param_count
            print("线路参数数量:", param_count)

            # 生成随机参数
            params = np.random.randn(param_count)

            # 创建量子虚拟机并运行线路
            qm = QMachine({n_qubits if n_qubits is not None else 3})
            qm.reset_states({n_qubits if n_qubits is not None else 3})
            ansatz(q_machine=qm, params=params)

            print("硬件高效ansatz线路构建完成")
            """

            return {
                "status": "success",
                "message": f"已生成VQC_HardwareEfficientAnsatz硬件高效ansatz代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet VQC_HardwareEfficientAnsatz API生成的真实代码，创建硬件高效的变分量子线路模板。"
            }
        elif "pyvqnet.qnn.vqc.VQC_AngleEmbedding" in tool_name or "VQC_AngleEmbedding" in tool_name:
            # VQC_AngleEmbedding 角度编码生成
            input_feat = arguments.get('input_feat')
            wires = arguments.get('wires')
            q_machine = arguments.get('q_machine', 'qm')
            rotation = arguments.get('rotation', '"X"')

            # 格式化参数
            input_feat_str = str(input_feat) if input_feat is not None else '[0.1, 0.2, 0.3]'
            wires_str = str(wires) if isinstance(wires, (list, tuple)) else '[0, 1, 2]'

            generated_code = f"""
            from pyvqnet.qnn.vqc import VQC_AngleEmbedding, QMachine, measure_all
            import numpy as np

            # 输入特征
            input_feat = {input_feat_str}
            wires = {wires_str}

            # 创建量子虚拟机
            qm = QMachine(len(wires))
            qm.reset_states(len(wires))

            # 应用角度编码
            VQC_AngleEmbedding(
                input_feat=input_feat,
                wires=wires,
                q_machine=qm,
                rotation={rotation}
            )

            # 测量
            result = measure_all(q_machine=qm, shots=1000)
            print("角度编码后测量结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成VQC_AngleEmbedding角度编码代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet VQC_AngleEmbedding API生成的真实代码，将经典特征编码到量子比特的旋转角度中。"
            }
        elif "pyvqnet.qnn.vqc.i" in tool_name:
            # i 门函数生成（恒等门函数版）
            q_machine = arguments.get('q_machine')
            wires = arguments.get('wires')
            params = arguments.get('params', None)
            use_dagger = arguments.get('use_dagger', False)

            params_str = params if params is not None else 'None'
            use_dagger_str = str(use_dagger) if isinstance(use_dagger, bool) else use_dagger

            generated_code = f"""
            from pyvqnet.qnn.vqc import i, QMachine

            # 应用Identity门（函数版）
            i(q_machine={q_machine}, wires={wires}, params={params_str}, use_dagger={use_dagger_str})

            # 打印应用后的状态
            print("应用Identity门后的状态:", {q_machine}.states)
            """

            return {
                "status": "success",
                "message": f"已生成Identity门（函数版）应用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet i API生成的真实代码。wires可以是单个比特索引或列表，params可选，use_dagger默认为False。"
            }
        elif "pyvqnet.qnn.vqc.t" in tool_name:
            # t 门函数生成（T相位门函数版）
            q_machine = arguments.get('q_machine')
            wires = arguments.get('wires')
            params = arguments.get('params', None)
            use_dagger = arguments.get('use_dagger', False)

            params_str = params if params is not None else 'None'
            use_dagger_str = str(use_dagger) if isinstance(use_dagger, bool) else use_dagger

            generated_code = f"""
            from pyvqnet.qnn.vqc import t, QMachine

            # 应用T门（π/4相位门）
            t(q_machine={q_machine}, wires={wires}, params={params_str}, use_dagger={use_dagger_str})

            # 打印应用后的状态
            print("应用T门后的状态:", {q_machine}.states)
            """

            return {
                "status": "success",
                "message": f"已生成T门（函数版）应用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet t API生成的真实代码。T门对量子态施加π/4的Z轴相位旋转，use_dagger=True时使用共轭版本（-π/4相位）。"
            }
        elif "pyvqnet.qnn.vqc.s" in tool_name:
            # s 门函数生成（S相位门函数版）
            q_machine = arguments.get('q_machine')
            wires = arguments.get('wires')
            params = arguments.get('params', None)
            use_dagger = arguments.get('use_dagger', False)

            params_str = params if params is not None else 'None'
            use_dagger_str = str(use_dagger) if isinstance(use_dagger, bool) else use_dagger

            generated_code = f"""
            from pyvqnet.qnn.vqc import s, QMachine

            # 应用S门（π/2相位门）
            s(q_machine={q_machine}, wires={wires}, params={params_str}, use_dagger={use_dagger_str})

            # 打印应用后的状态
            print("应用S门后的状态:", {q_machine}.states)
            """

            return {
                "status": "success",
                "message": f"已生成S门（函数版）应用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet s API生成的真实代码。S门对量子态施加π/2的Z轴相位旋转，use_dagger=True时使用共轭版本（-π/2相位）。"
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
        elif "pyvqnet.qnn.pq3.measure.ProbsMeasure" in tool_name:
            # ProbsMeasure 函数生成
            machine = arguments.get('machine', 'pq.CPUQVM()')
            prog = arguments.get('prog', 'pq.QProg()')
            measure_qubits = arguments.get('measure_qubits', [0])
            shots = arguments.get('shots', 1000)

            generated_code = f"""
            import pyqpanda3.core as pq
            from pyvqnet.qnn.pq3.measure import ProbsMeasure

            # 创建量子虚拟机和线路
            machine = {machine}
            prog = {prog}
            measure_qubits = {measure_qubits}
            shots = {shots}

            # 计算测量概率
            probabilities = ProbsMeasure(machine, prog, measure_qubits, shots=shots)
            print("测量概率结果:", probabilities)
            """

            return {
                "status": "success",
                "message": f"已生成ProbsMeasure概率测量代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet ProbsMeasure API生成的真实代码。shots=1时返回理论精确概率，shots>1时返回采样概率。"
            }
        elif "pyvqnet.qnn.vqc.CNOT" in tool_name:
            # CNOT 受控非门生成
            wires = arguments.get('wires', [0, 1])
            use_dagger = arguments.get('use_dagger', False)
            trainable = arguments.get('trainable', False)

            use_dagger_str = str(use_dagger) if isinstance(use_dagger, bool) else use_dagger
            trainable_str = str(trainable) if isinstance(trainable, bool) else trainable

            generated_code = f"""
            from pyvqnet.qnn.vqc import CNOT, QMachine

            # 应用CNOT门（控制位wires[0], 目标位wires[1]）
            cnot_gate = CNOT(wires={wires}, use_dagger={use_dagger_str}, trainable={trainable_str})
            # 应用到量子虚拟机（假设qm是已创建的QMachine实例）
            # cnot_gate(qm)
            """

            return {
                "status": "success",
                "message": f"已生成CNOT受控非门代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet CNOT API生成的真实代码。wires参数为[控制比特索引, 目标比特索引]。"
            }
        elif "pyvqnet.qnn.vqc.RX" in tool_name:
            # RX 旋转门生成
            wires = arguments.get('wires', 0)
            params = arguments.get('params', None)
            trainable = arguments.get('trainable', False)

            params_str = params if params is not None else 'None'
            trainable_str = str(trainable) if isinstance(trainable, bool) else trainable

            generated_code = f"""
            from pyvqnet.qnn.vqc import RX, QMachine

            # 应用RX旋转门
            rx_gate = RX(wires={wires}, params={params_str}, trainable={trainable_str})
            # 应用到量子虚拟机（假设qm是已创建的QMachine实例）
            # rx_gate(qm)
            """

            return {
                "status": "success",
                "message": f"已生成RX旋转门代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet RX API生成的真实代码。params为旋转角度参数，trainable=True时参数可训练。"
            }
        # ==================== QTensor 属性和方法 ====================
        elif "QTensor.shape" in tool_name or "shape" in tool_name:
            # QTensor.shape 属性
            generated_code = """
            from pyvqnet.tensor import QTensor
            import numpy as np

            # 创建张量
            tensor = QTensor(np.ones([2, 3, 4]))

            # 获取张量形状
            shape = tensor.shape
            print("张量形状:", shape)  # 输出: [2, 3, 4]
            """

            return {
                "status": "success",
                "message": f"已生成QTensor.shape使用代码",
                "generated_code": generated_code,
                "note": "QTensor.shape返回张量的维度列表，与numpy的shape属性功能类似。"
            }
        elif "QTensor.ndim" in tool_name or "ndim" in tool_name:
            # QTensor.ndim 属性
            generated_code = """
            from pyvqnet.tensor import QTensor
            import numpy as np

            # 创建张量
            tensor = QTensor(np.ones([2, 3, 4]))

            # 获取张量维度个数
            ndim = tensor.ndim
            print("张量维度个数:", ndim)  # 输出: 3
            """

            return {
                "status": "success",
                "message": f"已生成QTensor.ndim使用代码",
                "generated_code": generated_code,
                "note": "QTensor.ndim返回张量的维度数量，等价于len(tensor.shape)。"
            }
        elif "QTensor.size" in tool_name or "size" in tool_name and "QMachine" not in tool_name:
            # QTensor.size 属性
            generated_code = """
            from pyvqnet.tensor import QTensor
            import numpy as np

            # 创建张量
            tensor = QTensor(np.ones([2, 3]))

            # 获取张量元素总个数
            size = tensor.size
            print("张量元素总数:", size)  # 输出: 6
            """

            return {
                "status": "success",
                "message": f"已生成QTensor.size使用代码",
                "generated_code": generated_code,
                "note": "QTensor.size返回张量的元素总个数，等价于prod(tensor.shape)。"
            }
        elif "QTensor.numel" in tool_name or "numel" in tool_name:
            # QTensor.numel() 方法
            generated_code = """
            from pyvqnet.tensor import QTensor
            import numpy as np

            # 创建张量
            tensor = QTensor(np.ones([2, 3]))

            # 获取张量元素总个数
            numel = tensor.numel()
            print("张量元素总数:", numel)  # 输出: 6
            """

            return {
                "status": "success",
                "message": f"已生成QTensor.numel()使用代码",
                "generated_code": generated_code,
                "note": "QTensor.numel()方法与size属性功能完全相同，返回元素总个数。"
            }
        elif "QTensor.dtype" in tool_name or "dtype" in tool_name and "QuantumLayer" not in tool_name:
            # QTensor.dtype 属性
            generated_code = """
            from pyvqnet.tensor import QTensor
            from pyvqnet.dtype import *
            import numpy as np

            # 创建张量
            tensor = QTensor(np.ones([2, 3]), dtype=kfloat32)

            # 获取张量数据类型
            dtype = tensor.dtype
            print("张量数据类型:", dtype)  # 输出: kfloat32对应的数值
            """

            return {
                "status": "success",
                "message": f"已生成QTensor.dtype使用代码",
                "generated_code": generated_code,
                "note": "QTensor.dtype返回张量的数据类型，支持kfloat32、kfloat64、kcomplex64等类型。"
            }
        elif "QTensor.is_dense" in tool_name or "is_dense" in tool_name:
            # QTensor.is_dense 属性
            generated_code = """
            from pyvqnet.tensor import QTensor
            import numpy as np

            # 创建张量
            tensor = QTensor(np.ones([2, 3, 4]))

            # 检查是否是稠密张量
            is_dense = tensor.is_dense
            print("是否是稠密张量:", is_dense)  # 输出: 1
            """

            return {
                "status": "success",
                "message": f"已生成QTensor.is_dense使用代码",
                "generated_code": generated_code,
                "note": "QTensor.is_dense返回1表示是稠密张量，0表示是稀疏张量。"
            }
        elif "QTensor.is_contiguous" in tool_name or "is_contiguous" in tool_name:
            # QTensor.is_contiguous 属性
            generated_code = """
            from pyvqnet.tensor import QTensor
            import numpy as np

            # 创建张量
            tensor = QTensor(np.ones([2, 3, 4]))

            # 检查是否是连续存储
            is_contiguous = tensor.is_contiguous
            print("是否连续存储:", is_contiguous)  # 输出: True

            # 转置后不再连续
            transposed = tensor.permute((1, 0, 2))
            print("转置后是否连续:", transposed.is_contiguous)  # 输出: False
            """

            return {
                "status": "success",
                "message": f"已生成QTensor.is_contiguous使用代码",
                "generated_code": generated_code,
                "note": "QTensor.is_contiguous返回布尔值表示张量是否在内存中连续存储。"
            }
        elif "QTensor.zero_grad" in tool_name or "zero_grad" in tool_name:
            # QTensor.zero_grad() 方法
            generated_code = """
            from pyvqnet.tensor import QTensor
            import numpy as np

            # 创建需要计算梯度的张量
            tensor = QTensor([2, 3, 4, 5], requires_grad=True)

            # 将梯度清零
            tensor.zero_grad()
            print("清零后的梯度:", tensor.grad)  # 输出: [0., 0., 0., 0.]
            """

            return {
                "status": "success",
                "message": f"已生成QTensor.zero_grad()使用代码",
                "generated_code": generated_code,
                "note": "QTensor.zero_grad()方法将张量的梯度设置为零，通常在每次反向传播前调用。"
            }
        elif "QTensor.backward" in tool_name or "backward" in tool_name and "QuantumLayer" not in tool_name:
            # QTensor.backward() 方法
            generated_code = """
            from pyvqnet.tensor import QTensor
            import numpy as np

            # 创建需要计算梯度的张量
            target = QTensor([[0, 0, 1, 0, 0.2]], requires_grad=True)
            y = 2 * target + 3

            # 反向传播计算梯度
            y.backward()
            print("target的梯度:", target.grad)  # 输出: [[2. 2. 2. 2. 2.]]
            """

            return {
                "status": "success",
                "message": f"已生成QTensor.backward()使用代码",
                "generated_code": generated_code,
                "note": "QTensor.backward()方法执行反向传播，自动计算所有需要梯度的张量的梯度。可选参数grad是上游梯度，默认值为None。"
            }
        elif "QTensor.to_numpy" in tool_name or "to_numpy" in tool_name:
            # QTensor.to_numpy() 方法
            generated_code = """
            from pyvqnet.tensor import QTensor
            import numpy as np

            # 创建QTensor
            tensor = QTensor([2, 3, 4, 5], requires_grad=True)

            # 转换为numpy数组
            np_array = tensor.to_numpy()
            print("转换后的numpy数组:", np_array)  # 输出: [2. 3. 4. 5.]
            print("numpy数组类型:", type(np_array))  # 输出: <class 'numpy.ndarray'>
            """

            return {
                "status": "success",
                "message": f"已生成QTensor.to_numpy()使用代码",
                "generated_code": generated_code,
                "note": "QTensor.to_numpy()方法将张量数据转换为numpy数组。注意：bfloat16类型需要先转换为其他支持的类型再调用此方法。"
            }
        elif "QTensor.item" in tool_name or "item" in tool_name:
            # QTensor.item() 方法
            generated_code = """
            from pyvqnet.tensor import tensor, QTensor

            # 创建单元素张量
            t = tensor.ones([1])
            value = t.item()
            print("单元素张量的值:", value)  # 输出: 1.0
            print("值的类型:", type(value))  # 输出: <class 'float'>
            """

            return {
                "status": "success",
                "message": f"已生成QTensor.item()使用代码",
                "generated_code": generated_code,
                "note": "QTensor.item()方法仅适用于单元素张量，返回其Python标量值。"
            }
        elif "QTensor.contiguous" in tool_name or "contiguous" in tool_name and "is_contiguous" not in tool_name:
            # QTensor.contiguous() 方法
            generated_code = """
            from pyvqnet.tensor import QTensor
            import numpy as np

            # 创建张量并转置（非连续）
            tensor = QTensor(np.ones([2, 3]))
            transposed = tensor.permute((1, 0))
            print("转置后是否连续:", transposed.is_contiguous)  # 输出: False

            # 转换为连续存储
            contiguous_tensor = transposed.contiguous()
            print("转换后是否连续:", contiguous_tensor.is_contiguous)  # 输出: True
            """

            return {
                "status": "success",
                "message": f"已生成QTensor.contiguous()使用代码",
                "generated_code": generated_code,
                "note": "QTensor.contiguous()方法返回连续存储的张量副本，如果已经是连续的则返回自身。"
            }
        elif "QTensor.argmax" in tool_name or "argmax" in tool_name:
            # QTensor.argmax() 方法
            dim = arguments.get('dim', None)
            keepdims = arguments.get('keepdims', False)

            # 构建参数
            params = []
            if dim is not None:
                params.append(f'dim={dim}')
            if keepdims:
                params.append(f'keepdims={keepdims}')

            params_str = ", ".join(params) if params else ""

            generated_code = f"""
            from pyvqnet.tensor import QTensor
            import numpy as np

            # 创建测试张量
            a = QTensor([[1.3398, 0.2663, -0.2686, 0.2450],
                        [-0.7401, -0.8805, -0.3402, -1.1936],
                        [0.4907, -1.3948, -1.0691, -0.3132],
                        [-1.6092, 0.5419, -0.2993, 0.3195]])

            # 计算argmax
            max_indices = a.argmax({params_str})
            print("最大值索引:", max_indices)
            """

            return {
                "status": "success",
                "message": f"已生成QTensor.argmax()使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "QTensor.argmax()方法返回最大值的索引。dim参数指定计算的维度，keepdims指定是否保留维度。"
            }
        elif "QTensor.argmin" in tool_name or "argmin" in tool_name:
            # QTensor.argmin() 方法
            dim = arguments.get('dim', None)
            keepdims = arguments.get('keepdims', False)

            # 构建参数
            params = []
            if dim is not None:
                params.append(f'dim={dim}')
            if keepdims:
                params.append(f'keepdims={keepdims}')

            params_str = ", ".join(params) if params else ""

            generated_code = f"""
            from pyvqnet.tensor import QTensor
            import numpy as np

            # 创建测试张量
            a = QTensor([[1.3398, 0.2663, -0.2686, 0.2450],
                        [-0.7401, -0.8805, -0.3402, -1.1936],
                        [0.4907, -1.3948, -1.0691, -0.3132],
                        [-1.6092, 0.5419, -0.2993, 0.3195]])

            # 计算argmin
            min_indices = a.argmin({params_str})
            print("最小值索引:", min_indices)
            """

            return {
                "status": "success",
                "message": f"已生成QTensor.argmin()使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "QTensor.argmin()方法返回最小值的索引。dim参数指定计算的维度，keepdims指定是否保留维度。"
            }
        elif "QTensor.fill_" in tool_name or "fill_" in tool_name:
            # QTensor.fill_() 方法
            value = arguments.get('value', 0)

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor
            import numpy as np

            # 创建零张量
            shape = [2, 3]
            t = tensor.zeros(shape)

            # 填充指定值
            t.fill_({value})
            print("填充后的张量:", t)
            """

            return {
                "status": "success",
                "message": f"已生成QTensor.fill_()使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "QTensor.fill_()方法是原地操作，会修改原张量的数据，将所有元素设置为指定值。"
            }
        elif "QTensor.all" in tool_name or "all" in tool_name and "ProbsMeasure" not in tool_name:
            # QTensor.all() 方法
            generated_code = """
            from pyvqnet.tensor import tensor, QTensor
            import numpy as np

            # 创建全1张量
            t = tensor.ones([2, 3])
            all_non_zero = t.all()
            print("是否所有元素都非零:", all_non_zero)  # 输出: True

            # 修改一个元素为0
            t[0, 0] = 0
            all_non_zero = t.all()
            print("是否所有元素都非零:", all_non_zero)  # 输出: False
            """

            return {
                "status": "success",
                "message": f"已生成QTensor.all()使用代码",
                "generated_code": generated_code,
                "note": "QTensor.all()方法判断张量中是否所有元素都非零，返回布尔值。"
            }
        elif "QTensor.any" in tool_name or "any" in tool_name:
            # QTensor.any() 方法
            generated_code = """
            from pyvqnet.tensor import tensor, QTensor
            import numpy as np

            # 创建全0张量
            t = tensor.zeros([2, 3])
            any_non_zero = t.any()
            print("是否有非零元素:", any_non_zero)  # 输出: False

            # 修改一个元素为1
            t[0, 0] = 1
            any_non_zero = t.any()
            print("是否有非零元素:", any_non_zero)  # 输出: True
            """

            return {
                "status": "success",
                "message": f"已生成QTensor.any()使用代码",
                "generated_code": generated_code,
                "note": "QTensor.any()方法判断张量中是否存在至少一个非零元素，返回布尔值。"
            }
        elif "QTensor.fill_rand_binary_" in tool_name or "fill_rand_binary_" in tool_name:
            # QTensor.fill_rand_binary_() 方法
            threshold = arguments.get('v', 0.5)

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor
            import numpy as np

            # 创建张量
            a = np.arange(6).reshape(2, 3).astype(np.float32)
            t = QTensor(a)

            # 用二项分布随机值填充
            t.fill_rand_binary_({threshold})
            print("二项分布填充后的张量:", t)
            """

            return {
                "status": "success",
                "message": f"已生成QTensor.fill_rand_binary_()使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "QTensor.fill_rand_binary_()方法是原地操作，用二项分布随机值填充张量，大于阈值v的为1，否则为0。"
            }
        elif "QTensor.fill_rand_signed_uniform_" in tool_name or "fill_rand_signed_uniform_" in tool_name:
            # QTensor.fill_rand_signed_uniform_() 方法
            scale = arguments.get('v', 1)

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor
            import numpy as np

            # 创建张量
            a = np.arange(6).reshape(2, 3).astype(np.float32)
            t = QTensor(a)

            # 用有符号均匀分布随机值填充
            t.fill_rand_signed_uniform_({scale})
            print("有符号均匀分布填充后的张量:", t)
            """

            return {
                "status": "success",
                "message": f"已生成QTensor.fill_rand_signed_uniform_()使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "QTensor.fill_rand_signed_uniform_()方法是原地操作，用[-v, v]范围内的均匀分布随机值填充张量。"
            }
        elif "QTensor.fill_rand_uniform_" in tool_name or "fill_rand_uniform_" in tool_name:
            # QTensor.fill_rand_uniform_() 方法
            scale = arguments.get('v', 1)

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor
            import numpy as np

            # 创建张量
            a = np.arange(6).reshape(2, 3).astype(np.float32)
            t = QTensor(a)

            # 用均匀分布随机值填充
            t.fill_rand_uniform_({scale})
            print("均匀分布填充后的张量:", t)
            """

            return {
                "status": "success",
                "message": f"已生成QTensor.fill_rand_uniform_()使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "QTensor.fill_rand_uniform_()方法是原地操作，用[0, v]范围内的均匀分布随机值填充张量。"
            }
        elif "QTensor.fill_rand_normal_" in tool_name or "fill_rand_normal_" in tool_name:
            # QTensor.fill_rand_normal_() 方法
            mean = arguments.get('m', 0)
            std = arguments.get('s', 1)
            fast_math = arguments.get('fast_math', True)

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor
            import numpy as np

            # 创建张量
            a = np.arange(6).reshape(2, 3).astype(np.float32)
            t = QTensor(a)

            # 用正态分布随机值填充
            t.fill_rand_normal_(m={mean}, s={std}, fast_math={fast_math})
            print("正态分布填充后的张量:", t)
            """

            return {
                "status": "success",
                "message": f"已生成QTensor.fill_rand_normal_()使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "QTensor.fill_rand_normal_()方法是原地操作，用指定均值和标准差的正态分布随机值填充张量。"
            }
        # ==================== QTensor 全局函数 ====================
        elif "pyvqnet.tensor.diag" in tool_name or "diag" in tool_name and "fill_" not in tool_name:
            # diag 函数 - 构造对角矩阵
            t = arguments.get('t', 't')
            k = arguments.get('k', 0)

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor
            import numpy as np

            # 创建输入张量
            a = np.arange(16).reshape(4, 4).astype(np.float32)
            t = QTensor(a)

            # 构造对角矩阵/提取对角线元素
            result = tensor.diag({t}, k={k})
            print("diag结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成diag函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "diag函数：输入1D张量返回2D对角矩阵，输入2D张量返回1D对角线元素。k参数指定对角线偏移量。"
            }
        elif "pyvqnet.tensor.randu" in tool_name or "randu" in tool_name:
            # randu 函数 - 均匀分布随机张量
            shape = arguments.get('shape', '[2, 3]')
            min_val = arguments.get('min', 0.0)
            max_val = arguments.get('max', 1.0)

            generated_code = f"""
            from pyvqnet.tensor import tensor
            import pyvqnet

            # 创建均匀分布随机张量
            shape = {shape}
            t = tensor.randu(shape, min={min_val}, max={max_val})
            print("均匀分布随机张量:", t)
            """

            return {
                "status": "success",
                "message": f"已生成randu函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "randu函数创建[min, max)范围内的均匀分布随机张量，默认范围[0, 1)。"
            }
        elif "pyvqnet.tensor.randn" in tool_name or "randn" in tool_name:
            # randn 函数 - 正态分布随机张量
            shape = arguments.get('shape', '[2, 3]')
            mean = arguments.get('mean', 0.0)
            std = arguments.get('std', 1.0)

            generated_code = f"""
            from pyvqnet.tensor import tensor
            import pyvqnet

            # 创建正态分布随机张量
            shape = {shape}
            t = tensor.randn(shape, mean={mean}, std={std})
            print("正态分布随机张量:", t)
            """

            return {
                "status": "success",
                "message": f"已生成randn函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "randn函数创建指定均值和标准差的正态分布随机张量，默认mean=0, std=1。"
            }
        elif "pyvqnet.tensor.binomial" in tool_name or "binomial" in tool_name:
            # binomial 函数 - 二项分布
            total_counts = arguments.get('total_counts', 1000)
            probs = arguments.get('probs', 'tensor.randu([3,4])')

            generated_code = f"""
            from pyvqnet.tensor import tensor

            # 创建二项分布张量
            probs = {probs}
            result = tensor.binomial({total_counts}, probs)
            print("二项分布结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成binomial函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "binomial函数生成二项分布的随机张量，total_counts是试验次数，probs是事件概率。"
            }
        elif "pyvqnet.tensor.multinomial" in tool_name or "multinomial" in tool_name:
            # multinomial 函数 - 多项式分布采样
            t = arguments.get('t', 'tensor.QTensor([0.1, 10, 3, 1])')
            num_samples = arguments.get('num_samples', 3)

            generated_code = f"""
            from pyvqnet.tensor import tensor

            # 多项式分布采样
            weights = {t}
            indices = tensor.multinomial(weights, {num_samples})
            print("采样索引:", indices)
            """

            return {
                "status": "success",
                "message": f"已生成multinomial函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "multinomial函数根据输入的概率分布进行采样，返回采样的索引。"
            }
        elif "pyvqnet.tensor.triu" in tool_name or "triu" in tool_name:
            # triu 函数 - 上三角矩阵
            t = arguments.get('t', 't')
            diagonal = arguments.get('diagonal', 0)

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor
            import numpy as np

            # 创建输入张量
            a = np.arange(1.0, 2 * 6 * 5 + 1.0).reshape([2, 6, 5])
            t = QTensor(a)

            # 获取上三角矩阵
            upper_tri = tensor.triu({t}, diagonal={diagonal})
            print("上三角矩阵:", upper_tri)
            """

            return {
                "status": "success",
                "message": f"已生成triu函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "triu函数返回输入张量的上三角部分，其余元素置为0。diagonal参数指定对角线偏移量。"
            }
        elif "pyvqnet.tensor.tril" in tool_name or "tril" in tool_name:
            # tril 函数 - 下三角矩阵
            t = arguments.get('t', 't')
            diagonal = arguments.get('diagonal', 0)

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor
            import numpy as np

            # 创建输入张量
            a = np.arange(1.0, 12 * 5 + 1.0).reshape([12, 5])
            t = QTensor(a)

            # 获取下三角矩阵
            lower_tri = tensor.tril({t}, diagonal={diagonal})
            print("下三角矩阵:", lower_tri)
            """

            return {
                "status": "success",
                "message": f"已生成tril函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "tril函数返回输入张量的下三角部分，其余元素置为0。diagonal参数指定对角线偏移量。"
            }
        elif "pyvqnet.tensor.floor" in tool_name or "floor" in tool_name:
            # floor 函数 - 向下取整
            t = arguments.get('t', 't')

            generated_code = f"""
            from pyvqnet.tensor import tensor

            # 创建输入张量
            t = tensor.arange(-2.0, 2.0, 0.25)

            # 向下取整
            result = tensor.floor({t})
            print("向下取整结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成floor函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "floor函数对张量中的每个元素进行向下取整。"
            }
        elif "pyvqnet.tensor.ceil" in tool_name or "ceil" in tool_name:
            # ceil 函数 - 向上取整
            t = arguments.get('t', 't')

            generated_code = f"""
            from pyvqnet.tensor import tensor

            # 创建输入张量
            t = tensor.arange(-2.0, 2.0, 0.25)

            # 向上取整
            result = tensor.ceil({t})
            print("向上取整结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成ceil函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "ceil函数对张量中的每个元素进行向上取整。"
            }
        elif "pyvqnet.tensor.round" in tool_name or "round" in tool_name:
            # round 函数 - 四舍五入
            t = arguments.get('t', 't')

            generated_code = f"""
            from pyvqnet.tensor import tensor

            # 创建输入张量
            t = tensor.arange(-2.0, 2.0, 0.4)

            # 四舍五入
            result = tensor.round({t})
            print("四舍五入结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成round函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "round函数对张量中的每个元素进行四舍五入取整。"
            }
        elif "pyvqnet.tensor.argsort" in tool_name or "argsort" in tool_name:
            # argsort 函数 - 返回排序索引
            t = arguments.get('t', 't')
            axis = arguments.get('axis', 0)
            descending = arguments.get('descending', False)
            stable = arguments.get('stable', True)

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor
            import numpy as np

            # 创建输入张量
            a = np.random.randint(10, size=24).reshape(3,8).astype(np.float32)
            t = QTensor(a)

            # 获取排序后的索引
            sorted_indices = tensor.argsort({t}, axis={axis}, descending={descending}, stable={stable})
            print("排序索引:", sorted_indices)
            """

            return {
                "status": "success",
                "message": f"已生成argsort函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "argsort函数对张量排序并返回对应的索引，支持升序/降序和稳定排序。"
            }
        elif "pyvqnet.tensor.sort" in tool_name or "sort" in tool_name:
            # sort 函数 - 排序
            t = arguments.get('t', 't')
            axis = arguments.get('axis', 0)
            descending = arguments.get('descending', False)
            stable = arguments.get('stable', True)

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor
            import numpy as np

            # 创建输入张量
            a = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]]).astype(np.float32)
            t = QTensor(a)

            # 按指定轴排序
            sorted_tensor = tensor.sort({t}, axis={axis}, descending={descending}, stable={stable})
            print("排序结果:", sorted_tensor)
            """

            return {
                "status": "success",
                "message": f"已生成sort函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "sort函数按指定轴对张量进行排序，支持升序/降序和稳定排序。"
            }
            # argsort 函数 - 返回排序索引
            t = arguments.get('t', 't')
            axis = arguments.get('axis', 0)
            descending = arguments.get('descending', False)
            stable = arguments.get('stable', True)

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor
            import numpy as np

            # 创建输入张量
            a = np.random.randint(10, size=24).reshape(3,8).astype(np.float32)
            t = QTensor(a)

            # 获取排序后的索引
            sorted_indices = tensor.argsort({t}, axis={axis}, descending={descending}, stable={stable})
            print("排序索引:", sorted_indices)
            """

            return {
                "status": "success",
                "message": f"已生成argsort函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "argsort函数对张量排序并返回对应的索引，支持升序/降序和稳定排序。"
            }
        elif "pyvqnet.tensor.argtopK" in tool_name or "argtopK" in tool_name:
            # argtopK 函数 - 返回前K个元素的索引
            t = arguments.get('t', 't')
            k = arguments.get('k', 3)
            axis = arguments.get('axis', -1)
            if_descent = arguments.get('if_descent', True)

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入张量
            x = QTensor([
                24., 13., 15., 4., 3., 8., 11., 3., 6., 15., 24., 13., 15., 3., 3., 8., 7.,
                3., 6., 11.
            ])
            x = x.reshape([2, 5, 1, 2])

            # 获取前K个元素的索引
            topk_indices = tensor.argtopK({t if t != 't' else 'x'}, k={k}, axis={axis}, if_descent={if_descent})
            print("TopK索引:", topk_indices)
            """

            return {
                "status": "success",
                "message": f"已生成argtopK函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "argtopK函数返回沿指定维度的前K个最大/最小元素的索引。"
            }
        elif "pyvqnet.tensor.topK" in tool_name or "topK" in tool_name:
            # topK 函数 - 返回前K个元素
            t = arguments.get('t', 't')
            k = arguments.get('k', 3)
            axis = arguments.get('axis', -1)
            if_descent = arguments.get('if_descent', True)

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入张量
            x = QTensor([
                24., 13., 15., 4., 3., 8., 11., 3., 6., 15., 24., 13., 15., 3., 3., 8., 7.,
                3., 6., 11.
            ])
            x = x.reshape([2, 5, 1, 2])

            # 获取前K个元素
            topk_result = tensor.topK({t if t != 't' else 'x'}, k={k}, axis={axis}, if_descent={if_descent})
            print("TopK结果:", topk_result)
            """

            return {
                "status": "success",
                "message": f"已生成topK函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "topK函数返回沿指定维度的前K个最大/最小元素，if_descent=True返回最大元素，False返回最小元素。"
            }
            # argtopK 函数 - 返回前K个元素的索引
            t = arguments.get('t', 't')
            k = arguments.get('k', 3)
            axis = arguments.get('axis', -1)
            if_descent = arguments.get('if_descent', True)

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入张量
            x = QTensor([
                24., 13., 15., 4., 3., 8., 11., 3., 6., 15., 24., 13., 15., 3., 3., 8., 7.,
                3., 6., 11.
            ])
            x = x.reshape([2, 5, 1, 2])

            # 获取前K个元素的索引
            topk_indices = tensor.argtopK({t if t != 't' else 'x'}, k={k}, axis={axis}, if_descent={if_descent})
            print("TopK索引:", topk_indices)
            """

            return {
                "status": "success",
                "message": f"已生成argtopK函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "argtopK函数返回沿指定维度的前K个最大/最小元素的索引。"
            }
        elif "pyvqnet.tensor.add" in tool_name or ("add" in tool_name and "diag" not in tool_name and "pad" not in tool_name):
            # add 函数 - 元素相加
            t1 = arguments.get('t1', 't1')
            t2 = arguments.get('t2', 't2')

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入张量
            t1 = QTensor([1, 2, 3])
            t2 = QTensor([4, 5, 6])

            # 元素相加
            result = tensor.add({t1}, {t2})
            print("相加结果:", result)  # 等价于 t1 + t2
            """

            return {
                "status": "success",
                "message": f"已生成add函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "add函数实现两个张量按元素相加，等价于+运算符。"
            }
        elif "pyvqnet.tensor.sub" in tool_name or "sub" in tool_name:
            # sub 函数 - 元素相减
            t1 = arguments.get('t1', 't1')
            t2 = arguments.get('t2', 't2')

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入张量
            t1 = QTensor([1, 2, 3])
            t2 = QTensor([4, 5, 6])

            # 元素相减
            result = tensor.sub({t1}, {t2})
            print("相减结果:", result)  # 等价于 t1 - t2
            """

            return {
                "status": "success",
                "message": f"已生成sub函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "sub函数实现两个张量按元素相减，等价于-运算符。"
            }
        elif "pyvqnet.tensor.matmul" in tool_name or "matmul" in tool_name:
            # matmul 函数 - 矩阵乘法
            t1 = arguments.get('t1', 't1')
            t2 = arguments.get('t2', 't2')

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入矩阵
            t1 = tensor.ones([2, 3])
            t2 = tensor.ones([3, 4])

            # 矩阵乘法
            result = tensor.matmul({t1}, {t2})
            print("矩阵乘法结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成matmul函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "matmul函数支持矩阵乘法、批矩阵乘法、矩阵向量积和向量点积，等价于@运算符。"
            }
        elif "pyvqnet.tensor.mul" in tool_name or "mul" in tool_name:
            # mul 函数 - 元素相乘
            t1 = arguments.get('t1', 't1')
            t2 = arguments.get('t2', 't2')

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入张量
            t1 = QTensor([1, 2, 3])
            t2 = QTensor([4, 5, 6])

            # 元素相乘
            result = tensor.mul({t1}, {t2})
            print("相乘结果:", result)  # 等价于 t1 * t2
            """

            return {
                "status": "success",
                "message": f"已生成mul函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "mul函数实现两个张量按元素相乘，等价于*运算符。"
            }
        elif "pyvqnet.tensor.divide" in tool_name or "divide" in tool_name:
            # divide 函数 - 元素相除
            t1 = arguments.get('t1', 't1')
            t2 = arguments.get('t2', 't2')

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入张量
            t1 = QTensor([1, 2, 3])
            t2 = QTensor([4, 5, 6])

            # 元素相除
            result = tensor.divide({t1}, {t2})
            print("相除结果:", result)  # 等价于 t1 / t2
            """

            return {
                "status": "success",
                "message": f"已生成divide函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "divide函数实现两个张量按元素相除，等价于/运算符。"
            }
        elif "pyvqnet.tensor.sums" in tool_name or "sums" in tool_name:
            # sums 函数 - 按轴求和
            t = arguments.get('t', 't')
            axis = arguments.get('axis', None)
            keepdims = arguments.get('keepdims', False)

            # 处理None值
            axis_str = 'None' if axis is None else str(axis)

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入张量
            t = QTensor(([1, 2, 3], [4, 5, 6]))

            # 按轴求和
            result = tensor.sums({t}, axis={axis_str}, keepdims={keepdims})
            print("求和结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成sums函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "sums函数按指定轴对张量元素求和，axis=None时返回所有元素的总和。"
            }
        elif "pyvqnet.tensor.cumsum" in tool_name or "cumsum" in tool_name:
            # cumsum 函数 - 累积求和
            t = arguments.get('t', 't')
            axis = arguments.get('axis', -1)

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入张量
            t = QTensor(([1, 2, 3], [4, 5, 6]))

            # 累积求和
            result = tensor.cumsum({t}, axis={axis})
            print("累积求和结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成cumsum函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "cumsum函数沿指定轴计算元素的累积总和。"
            }
        elif "pyvqnet.tensor.mean" in tool_name or "mean" in tool_name:
            # mean 函数 - 按轴求平均
            t = arguments.get('t', 't')
            axis = arguments.get('axis', None)
            keepdims = arguments.get('keepdims', False)

            axis_str = 'None' if axis is None else str(axis)

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入张量
            t = QTensor([[1, 2, 3], [4, 5, 6.0]])

            # 按轴求平均
            result = tensor.mean({t}, axis={axis_str}, keepdims={keepdims})
            print("平均结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成mean函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "mean函数按指定轴对张量元素求平均值，axis=None时返回所有元素的平均值。"
            }
        elif "pyvqnet.tensor.median" in tool_name or "median" in tool_name:
            # median 函数 - 按轴求中位数
            t = arguments.get('t', 't')
            axis = arguments.get('axis', None)
            keepdims = arguments.get('keepdims', False)

            axis_str = 'None' if axis is None else str(axis)

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入张量
            t = QTensor([[1, 2, 3], [4, 5, 6.0]])

            # 按轴求中位数
            result = tensor.median({t}, axis={axis_str}, keepdims={keepdims})
            print("中位数结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成median函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "median函数按指定轴对张量元素求中位数，axis=None时返回所有元素的中位数。"
            }
        elif "pyvqnet.tensor.std" in tool_name or "std" in tool_name:
            # std 函数 - 计算标准差
            t = arguments.get('t', 't')
            axis = arguments.get('axis', None)
            keepdims = arguments.get('keepdims', False)
            unbiased = arguments.get('unbiased', True)

            axis_str = 'None' if axis is None else str(axis)

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入张量
            t = QTensor([[-0.8166, -1.3802, -0.3560]])

            # 计算标准差
            result = tensor.std({t}, axis={axis_str}, keepdims={keepdims}, unbiased={unbiased})
            print("标准差结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成std函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "std函数按指定轴计算张量元素的标准差，unbiased=True时使用贝塞尔修正。"
            }
        elif "pyvqnet.tensor.var" in tool_name or "var" in tool_name:
            # var 函数 - 计算方差
            t = arguments.get('t', 't')
            axis = arguments.get('axis', None)
            keepdims = arguments.get('keepdims', False)
            unbiased = arguments.get('unbiased', True)

            axis_str = 'None' if axis is None else str(axis)

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入张量
            t = QTensor([[-0.8166, -1.3802, -0.3560]])

            # 计算方差
            result = tensor.var({t}, axis={axis_str}, keepdims={keepdims}, unbiased={unbiased})
            print("方差结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成var函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "var函数按指定轴计算张量元素的方差，unbiased=True时使用贝塞尔修正。"
            }
        elif "pyvqnet.tensor.kron" in tool_name or "kron" in tool_name:
            # kron 函数 - Kronecker积
            t1 = arguments.get('t1', 't1')
            t2 = arguments.get('t2', 't2')

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入张量
            a = tensor.arange(1, 1+24).reshape([2, 1, 2, 3, 2])
            b = tensor.arange(1, 1+24).reshape([6, 4])

            # 计算Kronecker积
            result = tensor.kron({t1 if t1 != 't1' else 'a'}, {t2 if t2 != 't2' else 'b'})
            print("Kronecker积结果形状:", result.shape)
            """

            return {
                "status": "success",
                "message": f"已生成kron函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "kron函数计算两个张量的Kronecker积，支持多维张量。"
            }
        elif "pyvqnet.tensor.einsum" in tool_name or "einsum" in tool_name:
            # einsum 函数 - 爱因斯坦求和
            equation = arguments.get('equation', "'bn,anm,bm->ba'")
            operands = arguments.get('operands', ['vqnetl', 'vqneta', 'vqnetr'])

            operands_str = ", ".join(operands) if operands else ""

            generated_code = f"""
            from pyvqnet.tensor import tensor

            # 创建输入张量
            vqneta = tensor.randn((3, 5, 4))
            vqnetl = tensor.randn((2, 5))
            vqnetr = tensor.randn((2, 4))

            # 爱因斯坦求和
            result = tensor.einsum({equation}, {operands_str})
            print("einsum结果形状:", result.shape)
            """

            return {
                "status": "success",
                "message": f"已生成einsum函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "einsum函数使用爱因斯坦求和约定进行张量运算，支持复杂的张量操作。"
            }
        elif "pyvqnet.tensor.reciprocal" in tool_name or "reciprocal" in tool_name:
            # reciprocal 函数 - 倒数
            t = arguments.get('t', 't')

            generated_code = f"""
            from pyvqnet.tensor import tensor

            # 创建输入张量
            t = tensor.arange(1, 10, 1)

            # 计算倒数
            result = tensor.reciprocal({t})
            print("倒数结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成reciprocal函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "reciprocal函数计算张量每个元素的倒数。"
            }
        elif "pyvqnet.tensor.sign" in tool_name or "sign" in tool_name:
            # sign 函数 - 符号函数
            t = arguments.get('t', 't')

            generated_code = f"""
            from pyvqnet.tensor import tensor

            # 创建输入张量
            t = tensor.arange(-5, 5, 1)

            # 计算符号
            result = tensor.sign({t})
            print("符号结果:", result)  # -1表示负，0表示零，1表示正
            """

            return {
                "status": "success",
                "message": f"已生成sign函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "sign函数返回张量每个元素的符号：正为1，负为-1，零为0。"
            }
        elif "pyvqnet.tensor.neg" in tool_name or "neg" in tool_name:
            # neg 函数 - 取相反数
            t = arguments.get('t', 't')

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入张量
            t = QTensor([1, 2, 3])

            # 取相反数
            result = tensor.neg({t})
            print("相反数结果:", result)  # 等价于 -t
            """

            return {
                "status": "success",
                "message": f"已生成neg函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "neg函数计算张量每个元素的相反数，等价于负号运算符。"
            }
        elif "pyvqnet.tensor.trace" in tool_name or "trace" in tool_name:
            # trace 函数 - 矩阵的迹
            t = arguments.get('t', 't')
            k = arguments.get('k', 0)

            generated_code = f"""
            from pyvqnet.tensor import tensor

            # 创建输入矩阵
            t = tensor.randn([4, 4])

            # 计算矩阵的迹
            result = tensor.trace({t}, k={k})
            print("矩阵的迹:", result)
            """

            return {
                "status": "success",
                "message": f"已生成trace函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "trace函数计算二维矩阵对角线元素的和，k参数指定对角线偏移量。"
            }
        elif "pyvqnet.tensor.exp" in tool_name or "exp" in tool_name:
            # exp 函数 - 自然指数
            t = arguments.get('t', 't')

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入张量
            t = QTensor([1, 2, 3])

            # 计算自然指数
            result = tensor.exp({t})
            print("自然指数结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成exp函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "exp函数计算张量每个元素以e为底的指数值。"
            }
        elif "pyvqnet.tensor.acos" in tool_name or "acos" in tool_name:
            # acos 函数 - 反余弦
            t = arguments.get('t', 't')

            generated_code = f"""
            from pyvqnet.tensor import tensor
            import numpy as np

            # 创建输入张量
            a = np.arange(36).reshape(2,6,3).astype(np.float32) / 100
            t = tensor.QTensor(a)

            # 计算反余弦
            result = tensor.acos({t})
            print("反余弦结果形状:", result.shape)
            """

            return {
                "status": "success",
                "message": f"已生成acos函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "acos函数计算张量每个元素的反余弦值，输入范围应在[-1, 1]之间。"
            }
        elif "pyvqnet.tensor.asin" in tool_name or "asin" in tool_name:
            # asin 函数 - 反正弦
            t = arguments.get('t', 't')

            generated_code = f"""
            from pyvqnet.tensor import tensor

            # 创建输入张量
            t = tensor.arange(-1, 1, .5)

            # 计算反正弦
            result = tensor.asin({t})
            print("反正弦结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成asin函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "asin函数计算张量每个元素的反正弦值，输入范围应在[-1, 1]之间。"
            }
        elif "pyvqnet.tensor.atan" in tool_name or "atan" in tool_name:
            # atan 函数 - 反正切
            t = arguments.get('t', 't')

            generated_code = f"""
            from pyvqnet.tensor import tensor

            # 创建输入张量
            t = tensor.arange(-1, 1, .5)

            # 计算反正切
            result = tensor.atan({t})
            print("反正切结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成atan函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "atan函数计算张量每个元素的反正切值。"
            }
        elif "pyvqnet.tensor.tanh" in tool_name or "tanh" in tool_name:
            # tanh 函数 - 双曲正切
            t = arguments.get('t', 't')

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入张量
            t = QTensor([1, 2, 3])

            # 计算双曲正切
            result = tensor.tanh({t})
            print("双曲正切结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成tanh函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "tanh函数计算张量每个元素的双曲正切值，常用作激活函数。"
            }
        elif "pyvqnet.tensor.sinh" in tool_name or "sinh" in tool_name:
            # sinh 函数 - 双曲正弦
            t = arguments.get('t', 't')

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入张量
            t = QTensor([1, 2, 3])

            # 计算双曲正弦
            result = tensor.sinh({t})
            print("双曲正弦结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成sinh函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "sinh函数计算张量每个元素的双曲正弦值。"
            }
        elif "pyvqnet.tensor.cosh" in tool_name or "cosh" in tool_name:
            # cosh 函数 - 双曲余弦
            t = arguments.get('t', 't')

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入张量
            t = QTensor([1, 2, 3])

            # 计算双曲余弦
            result = tensor.cosh({t})
            print("双曲余弦结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成cosh函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "cosh函数计算张量每个元素的双曲余弦值。"
            }
        elif "pyvqnet.tensor.sin" in tool_name or "sin" in tool_name:
            # sin 函数 - 正弦
            t = arguments.get('t', 't')

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入张量
            t = QTensor([1, 2, 3])

            # 计算正弦
            result = tensor.sin({t})
            print("正弦结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成sin函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "sin函数计算张量每个元素的正弦值，输入为弧度。"
            }
        elif "pyvqnet.tensor.cos" in tool_name or "cos" in tool_name:
            # cos 函数 - 余弦
            t = arguments.get('t', 't')

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入张量
            t = QTensor([1, 2, 3])

            # 计算余弦
            result = tensor.cos({t})
            print("余弦结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成cos函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "cos函数计算张量每个元素的余弦值，输入为弧度。"
            }
        elif "pyvqnet.tensor.tan" in tool_name or "tan" in tool_name:
            # tan 函数 - 正切
            t = arguments.get('t', 't')

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入张量
            t = QTensor([1, 2, 3])

            # 计算正切
            result = tensor.tan({t})
            print("正切结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成tan函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "tan函数计算张量每个元素的正切值，输入为弧度。"
            }
        elif "pyvqnet.tensor.power" in tool_name or "power" in tool_name:
            # power 函数 - 幂运算
            t1 = arguments.get('t1', 't1')
            t2 = arguments.get('t2', 't2')

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入张量
            t1 = QTensor([1, 4, 3])
            t2 = QTensor([2, 5, 6])

            # 幂运算
            result = tensor.power({t1}, {t2})
            print("幂运算结果:", result)  # 等价于 t1 ** t2
            """

            return {
                "status": "success",
                "message": f"已生成power函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "power函数计算t1的t2次幂，按元素运算，等价于**运算符。"
            }
        elif "pyvqnet.tensor.abs" in tool_name or "abs" in tool_name:
            # abs 函数 - 绝对值
            t = arguments.get('t', 't')

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入张量
            t = QTensor([1, -2, 3])

            # 计算绝对值
            result = tensor.abs({t})
            print("绝对值结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成abs函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "abs函数计算张量每个元素的绝对值。"
            }
        elif "pyvqnet.tensor.log_softmax" in tool_name or "log_softmax" in tool_name:
            # log_softmax 函数 - 对数softmax
            t = arguments.get('t', 't')
            axis = arguments.get('axis', -1)

            generated_code = f"""
            from pyvqnet import tensor

            # 创建输入张量
            output = tensor.arange(1,13).reshape([3,2,2])

            # 计算log_softmax
            result = tensor.log_softmax({t if t != 't' else 'output'}, axis={axis})
            print("log_softmax结果形状:", result.shape)
            """

            return {
                "status": "success",
                "message": f"已生成log_softmax函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "log_softmax函数在指定轴上先计算softmax再取自然对数，常用于分类任务。"
            }
        elif "pyvqnet.tensor.log" in tool_name or "log" in tool_name:
            # log 函数 - 自然对数
            t = arguments.get('t', 't')

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入张量
            t = QTensor([1, 2, 3])

            # 计算自然对数
            result = tensor.log({t})
            print("自然对数结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成log函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "log函数计算张量每个元素的自然对数值。"
            }
        elif "pyvqnet.tensor.sqrt" in tool_name or "sqrt" in tool_name:
            # sqrt 函数 - 平方根
            t = arguments.get('t', 't')

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入张量
            t = QTensor([1, 2, 3])

            # 计算平方根
            result = tensor.sqrt({t})
            print("平方根结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成sqrt函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "sqrt函数计算张量每个元素的平方根值。"
            }
        elif "pyvqnet.tensor.square" in tool_name or "square" in tool_name:
            # square 函数 - 平方
            t = arguments.get('t', 't')

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入张量
            t = QTensor([1, 2, 3])

            # 计算平方
            result = tensor.square({t})
            print("平方结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成square函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "square函数计算张量每个元素的平方值，等价于t ** 2。"
            }
        elif "pyvqnet.tensor.eigh" in tool_name or "eigh" in tool_name:
            # eigh 函数 - 对称矩阵特征值分解
            t = arguments.get('t', 't')

            generated_code = f"""
            import numpy as np
            from pyvqnet import tensor

            # 生成随机对称矩阵
            def generate_random_symmetric_matrix(n):
                A = tensor.randn((n, n))
                A = A + A.transpose()
                return A

            n = 3
            symmetric_matrix = generate_random_symmetric_matrix(n)

            # 特征值分解
            evs, vecs = tensor.eigh({t if t != 't' else 'symmetric_matrix'})
            print("特征值:", evs)
            print("特征向量形状:", vecs.shape)
            """

            return {
                "status": "success",
                "message": f"已生成eigh函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "eigh函数计算实对称矩阵或复厄米矩阵的特征值和特征向量。"
            }
        elif "pyvqnet.tensor.frobenius_norm" in tool_name or "frobenius_norm" in tool_name:
            # frobenius_norm 函数 - F范数
            t = arguments.get('t', 't')
            axis = arguments.get('axis', None)
            keepdims = arguments.get('keepdims', False)

            axis_str = 'None' if axis is None else str(axis)

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入张量
            t = QTensor([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]],
                        [[13., 14., 15.], [16., 17., 18.]]])

            # 计算F范数
            result = tensor.frobenius_norm({t}, axis={axis_str}, keepdims={keepdims})
            print("F范数结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成frobenius_norm函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "frobenius_norm函数计算张量的Frobenius范数，支持按指定轴计算。"
            }
        elif "pyvqnet.tensor.maximum" in tool_name or "maximum" in tool_name:
            # maximum 函数 - 逐元素最大值
            t1 = arguments.get('t1', 't1')
            t2 = arguments.get('t2', 't2')

            generated_code = f"""
            from pyvqnet.tensor import tensor, QTensor

            # 创建输入张量
            t1 = QTensor([1, 3, 5])
            t2 = QTensor([2, 2, 6])

            # 逐元素取最大值
            result = tensor.maximum({t1}, {t2})
            print("逐元素最大值:", result)
            """

            return {
                "status": "success",
                "message": f"已生成maximum函数使用代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "maximum函数逐元素比较两个张量，返回较大值组成的新张量。"
            }
        # ==================== VQC 基础量子门 ====================
        elif "pyvqnet.qnn.vqc.RY" in tool_name or "ry" in tool_name:
            # RY 旋转门生成
            wires = arguments.get('wires', 0)
            params = arguments.get('params', None)
            trainable = arguments.get('trainable', False)

            params_str = params if params is not None else 'None'
            trainable_str = str(trainable) if isinstance(trainable, bool) else trainable

            generated_code = f"""
            from pyvqnet.qnn.vqc import RY, QMachine

            # 应用RY旋转门
            ry_gate = RY(wires={wires}, params={params_str}, trainable={trainable_str})
            # 应用到量子虚拟机（假设qm是已创建的QMachine实例）
            # ry_gate(qm)
            """

            return {
                "status": "success",
                "message": f"已生成RY旋转门代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet RY API生成的真实代码。params为绕Y轴的旋转角度参数。"
            }
        elif "pyvqnet.qnn.vqc.RZ" in tool_name or "rz" in tool_name:
            # RZ 旋转门生成
            wires = arguments.get('wires', 0)
            params = arguments.get('params', None)
            trainable = arguments.get('trainable', False)

            params_str = params if params is not None else 'None'
            trainable_str = str(trainable) if isinstance(trainable, bool) else trainable

            generated_code = f"""
            from pyvqnet.qnn.vqc import RZ, QMachine

            # 应用RZ旋转门
            rz_gate = RZ(wires={wires}, params={params_str}, trainable={trainable_str})
            # 应用到量子虚拟机（假设qm是已创建的QMachine实例）
            # rz_gate(qm)
            """

            return {
                "status": "success",
                "message": f"已生成RZ旋转门代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet RZ API生成的真实代码。params为绕Z轴的旋转角度参数。"
            }
        elif "pyvqnet.qnn.vqc.PauliX" in tool_name or "paulix" in tool_name:
            # PauliX 门生成
            wires = arguments.get('wires', 0)
            trainable = arguments.get('trainable', False)

            trainable_str = str(trainable) if isinstance(trainable, bool) else trainable

            generated_code = f"""
            from pyvqnet.qnn.vqc import PauliX, QMachine

            # 应用PauliX门（量子NOT门）
            paulix_gate = PauliX(wires={wires}, trainable={trainable_str})
            # 应用到量子虚拟机（假设qm是已创建的QMachine实例）
            # paulix_gate(qm)
            """

            return {
                "status": "success",
                "message": f"已生成PauliX门代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet PauliX API生成的真实代码，等价于经典逻辑的NOT门。"
            }
        elif "pyvqnet.qnn.vqc.PauliY" in tool_name or "pauliy" in tool_name:
            # PauliY 门生成
            wires = arguments.get('wires', 0)
            trainable = arguments.get('trainable', False)

            trainable_str = str(trainable) if isinstance(trainable, bool) else trainable

            generated_code = f"""
            from pyvqnet.qnn.vqc import PauliY, QMachine

            # 应用PauliY门
            pauliy_gate = PauliY(wires={wires}, trainable={trainable_str})
            # 应用到量子虚拟机（假设qm是已创建的QMachine实例）
            # pauliy_gate(qm)
            """

            return {
                "status": "success",
                "message": f"已生成PauliY门代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet PauliY API生成的真实代码。"
            }
        elif "pyvqnet.qnn.vqc.I" in tool_name or "identity" in tool_name or "i" in tool_name:
            # Identity 门生成
            wires = arguments.get('wires', 0)
            trainable = arguments.get('trainable', False)

            trainable_str = str(trainable) if isinstance(trainable, bool) else trainable

            generated_code = f"""
            from pyvqnet.qnn.vqc import I, QMachine

            # 应用Identity门（恒等门）
            identity_gate = I(wires={wires}, trainable={trainable_str})
            # 应用到量子虚拟机（假设qm是已创建的QMachine实例）
            # identity_gate(qm)
            """

            return {
                "status": "success",
                "message": f"已生成Identity门代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet Identity API生成的真实代码，不对量子态做任何改变，常用于占位或噪声模拟。"
            }
        elif "pyvqnet.qnn.vqc.T" in tool_name or "t_gate" in tool_name or "t" in tool_name:
            # T 门生成（π/4相位门）
            wires = arguments.get('wires', 0)
            trainable = arguments.get('trainable', False)
            use_dagger = arguments.get('use_dagger', False)

            trainable_str = str(trainable) if isinstance(trainable, bool) else trainable
            use_dagger_str = str(use_dagger) if isinstance(use_dagger, bool) else use_dagger

            generated_code = f"""
            from pyvqnet.qnn.vqc import T, QMachine

            # 应用T门（π/4相位门）
            t_gate = T(wires={wires}, trainable={trainable_str}, use_dagger={use_dagger_str})
            # 应用到量子虚拟机（假设qm是已创建的QMachine实例）
            # t_gate(qm)
            """

            return {
                "status": "success",
                "message": f"已生成T门代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet T API生成的真实代码。T门对量子态施加π/4的Z轴相位旋转，use_dagger=True时使用共轭版本（-π/4相位）。"
            }
        elif "pyvqnet.qnn.vqc.S" in tool_name or "s_gate" in tool_name or "s" in tool_name:
            # S 门生成（π/2相位门）
            wires = arguments.get('wires', 0)
            trainable = arguments.get('trainable', False)
            use_dagger = arguments.get('use_dagger', False)

            trainable_str = str(trainable) if isinstance(trainable, bool) else trainable
            use_dagger_str = str(use_dagger) if isinstance(use_dagger, bool) else use_dagger

            generated_code = f"""
            from pyvqnet.qnn.vqc import S, QMachine

            # 应用S门（π/2相位门）
            s_gate = S(wires={wires}, trainable={trainable_str}, use_dagger={use_dagger_str})
            # 应用到量子虚拟机（假设qm是已创建的QMachine实例）
            # s_gate(qm)
            """

            return {
                "status": "success",
                "message": f"已生成S门代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet S API生成的真实代码。S门对量子态施加π/2的Z轴相位旋转，use_dagger=True时使用共轭版本（-π/2相位）。"
            }
        elif "pyvqnet.qnn.vqc.PauliZ" in tool_name or "pauliz" in tool_name:
            # PauliZ 门生成
            wires = arguments.get('wires', 0)
            trainable = arguments.get('trainable', False)

            trainable_str = str(trainable) if isinstance(trainable, bool) else trainable

            generated_code = f"""
            from pyvqnet.qnn.vqc import PauliZ, QMachine

            # 应用PauliZ门（相位翻转门）
            pauliz_gate = PauliZ(wires={wires}, trainable={trainable_str})
            # 应用到量子虚拟机（假设qm是已创建的QMachine实例）
            # pauliz_gate(qm)
            """

            return {
                "status": "success",
                "message": f"已生成PauliZ门代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet PauliZ API生成的真实代码，用于翻转量子态相位。"
            }
        # ==================== 新增: nn模块 神经网络层 ====================
        elif "pyvqnet.nn.Linear" in tool_name or "Linear" in tool_name:
            # Linear 全连接层生成
            input_channels = arguments.get('input_channels')
            output_channels = arguments.get('output_channels')
            use_bias = arguments.get('use_bias', True)
            dtype = arguments.get('dtype', None)
            name = arguments.get('name', "")

            # 格式化参数
            use_bias_str = str(use_bias) if isinstance(use_bias, bool) else use_bias
            dtype_str = dtype if dtype is not None else 'None'
            name_str = f'"{name}"' if isinstance(name, str) else name

            generated_code = f"""
            import numpy as np
            from pyvqnet.nn import Linear
            from pyvqnet.tensor import QTensor

            # 创建全连接层
            linear_layer = Linear(
                input_channels={input_channels},
                output_channels={output_channels},
                use_bias={use_bias_str},
                dtype={dtype_str},
                name={name_str}
            )

            # 示例输入
            input = QTensor(np.random.randn(2, {input_channels}), requires_grad=True)

            # 前向传播
            output = linear_layer(input)
            print("全连接层输出形状:", output.shape)
            print("全连接层输出:", output)
            """

            return {
                "status": "success",
                "message": f"已生成Linear全连接层代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet Linear API生成的真实代码，实现y = x@A.T + b的线性变换。"
            }
        elif "pyvqnet.nn.Conv2d" in tool_name or "Conv2d" in tool_name:
            # Conv2d 二维卷积层生成
            in_channels = arguments.get('in_channels')
            out_channels = arguments.get('out_channels')
            kernel_size = arguments.get('kernel_size')
            stride = arguments.get('stride', 1)
            padding = arguments.get('padding', 0)
            dilation = arguments.get('dilation', 1)
            groups = arguments.get('groups', 1)
            use_bias = arguments.get('use_bias', True)
            dtype = arguments.get('dtype', None)
            name = arguments.get('name', "")

            # 格式化参数
            stride_str = str(stride) if isinstance(stride, int) else stride
            padding_str = str(padding) if isinstance(padding, int) else padding
            dilation_str = str(dilation) if isinstance(dilation, int) else dilation
            groups_str = str(groups) if isinstance(groups, int) else groups
            use_bias_str = str(use_bias) if isinstance(use_bias, bool) else use_bias
            dtype_str = dtype if dtype is not None else 'None'
            name_str = f'"{name}"' if isinstance(name, str) else name

            generated_code = f"""
            import numpy as np
            from pyvqnet.nn import Conv2d
            from pyvqnet.tensor import QTensor

            # 创建二维卷积层
            conv_layer = Conv2d(
                in_channels={in_channels},
                out_channels={out_channels},
                kernel_size={kernel_size},
                stride={stride_str},
                padding={padding_str},
                dilation={dilation_str},
                groups={groups_str},
                use_bias={use_bias_str},
                dtype={dtype_str},
                name={name_str}
            )

            # 示例输入 (batch_size, channels, height, width)
            input = QTensor(np.random.randn(2, {in_channels}, 32, 32), requires_grad=True)

            # 前向传播
            output = conv_layer(input)
            print("卷积层输出形状:", output.shape)
            print("卷积层输出:", output)
            """

            return {
                "status": "success",
                "message": f"已生成Conv2d二维卷积层代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet Conv2d API生成的真实代码，实现二维卷积操作。"
            }
        elif "pyvqnet.nn.BatchNorm2d" in tool_name or "BatchNorm2d" in tool_name:
            # BatchNorm2d 二维批量归一化层生成
            channel_num = arguments.get('channel_num')
            momentum = arguments.get('momentum', 0.1)
            epsilon = arguments.get('epsilon', 1e-5)
            affine = arguments.get('affine', True)
            dtype = arguments.get('dtype', None)
            name = arguments.get('name', "")

            # 格式化参数
            momentum_str = str(momentum) if isinstance(momentum, (int, float)) else momentum
            epsilon_str = str(epsilon) if isinstance(epsilon, (int, float)) else epsilon
            affine_str = str(affine) if isinstance(affine, bool) else affine
            dtype_str = dtype if dtype is not None else 'None'
            name_str = f'"{name}"' if isinstance(name, str) else name

            generated_code = f"""
            import numpy as np
            from pyvqnet.nn import BatchNorm2d
            from pyvqnet.tensor import QTensor

            # 创建二维批量归一化层
            bn_layer = BatchNorm2d(
                channel_num={channel_num},
                momentum={momentum_str},
                epsilon={epsilon_str},
                affine={affine_str},
                dtype={dtype_str},
                name={name_str}
            )

            # 示例输入 (batch_size, channels, height, width)
            input = QTensor(np.random.randn(2, {channel_num}, 32, 32), requires_grad=True)

            # 前向传播
            bn_layer.train()  # 设置为训练模式
            output = bn_layer(input)
            print("批量归一化层输出形状:", output.shape)
            print("批量归一化层输出:", output)
            """

            return {
                "status": "success",
                "message": f"已生成BatchNorm2d批量归一化层代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet BatchNorm2d API生成的真实代码，对四维张量进行批量归一化操作。"
            }
        elif "pyvqnet.nn.dropout.Dropout" in tool_name or "Dropout" in tool_name:
            # Dropout 层生成
            dropout_rate = arguments.get('dropout_rate', 0.5)
            name = arguments.get('name', "")

            # 格式化参数
            dropout_rate_str = str(dropout_rate) if isinstance(dropout_rate, (int, float)) else dropout_rate
            name_str = f'"{name}"' if isinstance(name, str) else name

            generated_code = f"""
            import numpy as np
            from pyvqnet.nn.dropout import Dropout
            from pyvqnet.tensor import QTensor

            # 创建Dropout层
            dropout_layer = Dropout(
                dropout_rate={dropout_rate_str},
                name={name_str}
            )

            # 示例输入 (batch_size, channels, height, width)
            input = QTensor(np.random.randn(2, 3, 32, 32), requires_grad=True)

            # 前向传播
            dropout_layer.train()  # 设置为训练模式
            output = dropout_layer(input)
            print("Dropout层输出形状:", output.shape)
            print("Dropout层输出:", output)
            """

            return {
                "status": "success",
                "message": f"已生成Dropout层代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet Dropout API生成的真实代码，随机将一些单元的输出设置为零，防止过拟合。"
            }
        elif "pyvqnet.nn.AvgPool2D" in tool_name or "AvgPool2D" in tool_name:
            # AvgPool2D 二维平均池化层生成
            kernel = arguments.get('kernel')
            stride = arguments.get('stride')
            padding = arguments.get('padding', "valid")
            name = arguments.get('name', "")

            # 格式化参数
            kernel_str = str(kernel) if isinstance(kernel, (int, list, tuple)) else kernel
            stride_str = str(stride) if isinstance(stride, (int, list, tuple)) else stride
            padding_str = f'"{padding}"' if isinstance(padding, str) else str(padding)
            name_str = f'"{name}"' if isinstance(name, str) else name

            generated_code = f"""
            import numpy as np
            from pyvqnet.nn import AvgPool2D
            from pyvqnet.tensor import QTensor

            # 创建二维平均池化层
            avgpool_layer = AvgPool2D(
                kernel={kernel_str},
                stride={stride_str},
                padding={padding_str},
                name={name_str}
            )

            # 示例输入 (batch_size, channels, height, width)
            input = QTensor(np.random.randn(2, 3, 32, 32), requires_grad=True)

            # 前向传播
            output = avgpool_layer(input)
            print("平均池化层输出形状:", output.shape)
            print("平均池化层输出:", output)
            """

            return {
                "status": "success",
                "message": f"已生成AvgPool2D二维平均池化层代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet AvgPool2D API生成的真实代码，对二维输入进行平均池化操作。"
            }
        elif "pyvqnet.nn.MaxPool2D" in tool_name or "MaxPool2D" in tool_name:
            # MaxPool2D 二维最大池化层生成
            kernel = arguments.get('kernel')
            stride = arguments.get('stride')
            padding = arguments.get('padding', "valid")
            name = arguments.get('name', "")

            # 格式化参数
            kernel_str = str(kernel) if isinstance(kernel, (int, list, tuple)) else kernel
            stride_str = str(stride) if isinstance(stride, (int, list, tuple)) else stride
            padding_str = f'"{padding}"' if isinstance(padding, str) else str(padding)
            name_str = f'"{name}"' if isinstance(name, str) else name

            generated_code = f"""
            import numpy as np
            from pyvqnet.nn import MaxPool2D
            from pyvqnet.tensor import QTensor

            # 创建二维最大池化层
            maxpool_layer = MaxPool2D(
                kernel={kernel_str},
                stride={stride_str},
                padding={padding_str},
                name={name_str}
            )

            # 示例输入 (batch_size, channels, height, width)
            input = QTensor(np.random.randn(2, 3, 32, 32), requires_grad=True)

            # 前向传播
            output = maxpool_layer(input)
            print("最大池化层输出形状:", output.shape)
            print("最大池化层输出:", output)
            """

            return {
                "status": "success",
                "message": f"已生成MaxPool2D二维最大池化层代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet MaxPool2D API生成的真实代码，对二维输入进行最大池化操作。"
            }
        # ==================== 新增: nn模块 激活函数 ====================
        elif "pyvqnet.nn.ReLu" in tool_name or "ReLu" in tool_name:
            # ReLu 激活函数生成
            name = arguments.get('name', "")

            name_str = f'"{name}"' if isinstance(name, str) else name

            generated_code = f"""
            import numpy as np
            from pyvqnet.nn import ReLu
            from pyvqnet.tensor import QTensor

            # 创建ReLU激活函数
            relu = ReLu(name={name_str})

            # 示例输入
            input = QTensor(np.array([-2, -1, 0, 1, 2]), requires_grad=True)

            # 前向传播
            output = relu(input)
            print("ReLU激活输出:", output)
            """

            return {
                "status": "success",
                "message": f"已生成ReLU激活函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet ReLu API生成的真实代码，实现max(0, x)的激活函数。"
            }
        elif "pyvqnet.nn.Gelu" in tool_name or "Gelu" in tool_name:
            # Gelu 激活函数生成
            approximate = arguments.get('approximate', "tanh")
            name = arguments.get('name', "")

            approximate_str = f'"{approximate}"' if isinstance(approximate, str) else approximate
            name_str = f'"{name}"' if isinstance(name, str) else name

            generated_code = f"""
            import numpy as np
            from pyvqnet.nn import Gelu
            from pyvqnet.tensor import QTensor

            # 创建GELU激活函数
            gelu = Gelu(approximate={approximate_str}, name={name_str})

            # 示例输入
            input = QTensor(np.array([-2, -1, 0, 1, 2]), requires_grad=True)

            # 前向传播
            output = gelu(input)
            print("GELU激活输出:", output)
            """

            return {
                "status": "success",
                "message": f"已生成GELU激活函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet Gelu API生成的真实代码，实现高斯误差线性单元激活函数。"
            }
        elif "pyvqnet.nn.Sigmoid" in tool_name or "Sigmoid" in tool_name:
            # Sigmoid 激活函数生成
            name = arguments.get('name', "")

            name_str = f'"{name}"' if isinstance(name, str) else name

            generated_code = f"""
            import numpy as np
            from pyvqnet.nn import Sigmoid
            from pyvqnet.tensor import QTensor

            # 创建Sigmoid激活函数
            sigmoid = Sigmoid(name={name_str})

            # 示例输入
            input = QTensor(np.array([-5, -2, 0, 2, 5]), requires_grad=True)

            # 前向传播
            output = sigmoid(input)
            print("Sigmoid激活输出:", output)
            """

            return {
                "status": "success",
                "message": f"已生成Sigmoid激活函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet Sigmoid API生成的真实代码，实现1/(1+exp(-x))的S型激活函数。"
            }
        # ==================== 新增: nn模块 损失函数 ====================
        elif "pyvqnet.nn.CrossEntropyLoss" in tool_name or "CrossEntropyLoss" in tool_name:
            # CrossEntropyLoss 交叉熵损失函数生成
            name = arguments.get('name', "")

            name_str = f'"{name}"' if isinstance(name, str) else name

            generated_code = f"""
            import numpy as np
            from pyvqnet.nn import CrossEntropyLoss
            from pyvqnet.tensor import QTensor

            # 创建交叉熵损失函数
            loss_fn = CrossEntropyLoss(name={name_str})

            # 示例输入 (batch_size, num_classes)
            input = QTensor(np.random.randn(3, 5), requires_grad=True)
            # 示例目标 (batch_size,)
            target = QTensor(np.array([0, 2, 1]), dtype='int64')

            # 计算损失
            loss = loss_fn(input, target)
            print("交叉熵损失:", loss)

            # 反向传播
            loss.backward()
            print("输入梯度:", input.grad)
            """

            return {
                "status": "success",
                "message": f"已生成CrossEntropyLoss损失函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet CrossEntropyLoss API生成的真实代码，计算输入和目标之间的交叉熵损失。"
            }
        elif "pyvqnet.nn.NLL_Loss" in tool_name or "NLL_Loss" in tool_name:
            # NLL_Loss 负对数似然损失函数生成
            name = arguments.get('name', "")

            name_str = f'"{name}"' if isinstance(name, str) else name

            generated_code = f"""
            import numpy as np
            from pyvqnet.nn import NLL_Loss, LogSoftmax
            from pyvqnet.tensor import QTensor

            # 创建负对数似然损失函数
            loss_fn = NLL_Loss(name={name_str})
            log_softmax = LogSoftmax(dim=1)

            # 示例输入 (batch_size, num_classes)
            input = QTensor(np.random.randn(3, 5), requires_grad=True)
            # 经过LogSoftmax处理
            log_probs = log_softmax(input)
            # 示例目标 (batch_size,)
            target = QTensor(np.array([0, 2, 1]), dtype='int64')

            # 计算损失
            loss = loss_fn(log_probs, target)
            print("负对数似然损失:", loss)

            # 反向传播
            loss.backward()
            print("输入梯度:", input.grad)
            """

            return {
                "status": "success",
                "message": f"已生成NLL_Loss损失函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet NLL_Loss API生成的真实代码，计算对数概率和目标之间的负对数似然损失。"
            }
        # ==================== 新增: optim模块 优化器 ====================
        elif "pyvqnet.optim.adam.Adam" in tool_name or "Adam" in tool_name:
            # Adam 优化器生成
            params = arguments.get('params', 'model.parameters()')
            lr = arguments.get('lr', 0.01)
            beta1 = arguments.get('beta1', 0.9)
            beta2 = arguments.get('beta2', 0.999)
            epsilon = arguments.get('epsilon', 1e-8)
            weight_decay = arguments.get('weight_decay', 0)
            amsgrad = arguments.get('amsgrad', False)

            # 格式化参数
            lr_str = str(lr) if isinstance(lr, (int, float)) else lr
            beta1_str = str(beta1) if isinstance(beta1, (int, float)) else beta1
            beta2_str = str(beta2) if isinstance(beta2, (int, float)) else beta2
            epsilon_str = str(epsilon) if isinstance(epsilon, (int, float)) else epsilon
            weight_decay_str = str(weight_decay) if isinstance(weight_decay, (int, float)) else weight_decay
            amsgrad_str = str(amsgrad) if isinstance(amsgrad, bool) else amsgrad

            generated_code = f"""
            from pyvqnet.nn import Linear, Module
            from pyvqnet.optim import Adam
            from pyvqnet.tensor import QTensor
            import numpy as np

            # 示例模型
            class SimpleModel(Module):
                def __init__(self):
                    super().__init__()
                    self.linear = Linear(10, 2)

                def forward(self, x):
                    return self.linear(x)

            model = SimpleModel()

            # 创建Adam优化器
            optimizer = Adam(
                params={params},
                lr={lr_str},
                beta1={beta1_str},
                beta2={beta2_str},
                epsilon={epsilon_str},
                weight_decay={weight_decay_str},
                amsgrad={amsgrad_str}
            )

            # 示例训练步骤
            input = QTensor(np.random.randn(3, 10), requires_grad=True)
            target = QTensor(np.array([0, 1, 0]), dtype='int64')

            # 前向传播
            output = model(input)
            loss = output.sum()

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()
            optimizer.zero_grad()

            print("优化器更新完成")
            """

            return {
                "status": "success",
                "message": f"已生成Adam优化器代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet Adam API生成的真实代码，实现自适应矩估计优化算法。"
            }
        elif "pyvqnet.optim.sgd.SGD" in tool_name or "SGD" in tool_name:
            # SGD 优化器生成
            params = arguments.get('params', 'model.parameters()')
            lr = arguments.get('lr', 0.01)
            momentum = arguments.get('momentum', 0)
            nesterov = arguments.get('nesterov', False)

            # 格式化参数
            lr_str = str(lr) if isinstance(lr, (int, float)) else lr
            momentum_str = str(momentum) if isinstance(momentum, (int, float)) else momentum
            nesterov_str = str(nesterov) if isinstance(nesterov, bool) else nesterov

            generated_code = f"""
            from pyvqnet.nn import Linear, Module
            from pyvqnet.optim import SGD
            from pyvqnet.tensor import QTensor
            import numpy as np

            # 示例模型
            class SimpleModel(Module):
                def __init__(self):
                    super().__init__()
                    self.linear = Linear(10, 2)

                def forward(self, x):
                    return self.linear(x)

            model = SimpleModel()

            # 创建SGD优化器
            optimizer = SGD(
                params={params},
                lr={lr_str},
                momentum={momentum_str},
                nesterov={nesterov_str}
            )

            # 示例训练步骤
            input = QTensor(np.random.randn(3, 10), requires_grad=True)
            target = QTensor(np.array([0, 1, 0]), dtype='int64')

            # 前向传播
            output = model(input)
            loss = output.sum()

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()
            optimizer.zero_grad()

            print("优化器更新完成")
            """

            return {
                "status": "success",
                "message": f"已生成SGD优化器代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet SGD API生成的真实代码，实现随机梯度下降优化算法。"
            }
        # ==================== 新增: vqc模块 测量函数 ====================
        elif "pyvqnet.qnn.vqc.Samples" in tool_name or "Samples" in tool_name:
            # Samples 采样测量生成
            wires = arguments.get('wires', None)
            obs = arguments.get('obs', None)
            shots = arguments.get('shots', 1)
            name = arguments.get('name', "")

            # 格式化参数
            wires_str = 'None' if wires is None else str(wires)
            obs_str = 'None' if obs is None else str(obs)
            shots_str = str(shots) if isinstance(shots, int) else shots
            name_str = f'"{name}"' if isinstance(name, str) else name

            generated_code = f"""
            from pyvqnet.qnn.vqc import Samples, QMachine, hadamard
            from pyvqnet.tensor import QTensor

            # 创建量子虚拟机
            qm = QMachine(2)
            qm.reset_states(1)

            # 制备量子态
            hadamard(q_machine=qm, wires=0)
            hadamard(q_machine=qm, wires=1)

            # 创建采样测量
            measure = Samples(
                wires={wires_str},
                obs={obs_str},
                shots={shots_str},
                name={name_str}
            )

            # 执行测量
            samples = measure(q_machine=qm)
            print("采样结果:", samples)
            """

            return {
                "status": "success",
                "message": f"已生成Samples采样测量代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet Samples API生成的真实代码，对量子态进行多次采样测量。"
            }
        elif "pyvqnet.qnn.vqc.HermitianExpval" in tool_name or "HermitianExpval" in tool_name:
            # HermitianExpval 厄米特期望值测量生成
            obs = arguments.get('obs')
            name = arguments.get('name', "")

            name_str = f'"{name}"' if isinstance(name, str) else name

            generated_code = f"""
            from pyvqnet.qnn.vqc import HermitianExpval, QMachine, PauliX, hadamard
            from pyvqnet.tensor import QTensor
            import numpy as np

            # 创建量子虚拟机
            qm = QMachine(1)
            qm.reset_states(1)

            # 制备量子态
            hadamard(q_machine=qm, wires=0)

            # 定义可观测量（泡利X矩阵）
            obs = np.array([[0, 1], [1, 0]], dtype=np.complex64)

            # 创建厄米特期望值测量
            measure = HermitianExpval(
                obs={obs if obs is not None else 'obs'},
                name={name_str}
            )

            # 执行测量
            expval = measure(q_machine=qm)
            print("期望值:", expval)
            """

            return {
                "status": "success",
                "message": f"已生成HermitianExpval期望值测量代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet HermitianExpval API生成的真实代码，计算厄米特可观测量的期望值。"
            }
        elif "pyvqnet.qnn.pq3.quantumlayer.QuantumLayerAdjoint" in tool_name or "QuantumLayerAdjoint" in tool_name:
            # QuantumLayerAdjoint 伴随法梯度计算层
            pq3_vqc_circuit = arguments.get('pq3_vqc_circuit')
            param_num = arguments.get('param_num')
            pauli_dicts = arguments.get('pauli_dicts')
            name = arguments.get('name', "")

            name_str = f'"{name}"' if isinstance(name, str) else name

            generated_code = f"""
            from pyvqnet.qnn.pq3 import QuantumLayerAdjoint
            from pyvqnet import tensor
            from pyqpanda3.vqcircuit import VQCircuit
            import pyqpanda3 as pq3

            # 使用VQCircuit接口的量子线路函数
            def pqctest(x, param):
                vqc = VQCircuit()
                vqc.set_Param([len(param) + len(x)])
                w_offset = len(x)
                for j in range(len(x)):
                    vqc << pq3.core.RX(j, vqc.Param([j]))
                return vqc

            # 创建伴随法梯度计算层
            n = 7
            Xn_string = ' '.join([f'X{{i}}' for i in range(n)])
            pauli_dict = {{Xn_string: 1.}}

            layer = QuantumLayerAdjoint(pqctest, param_num={param_num}, pauli_dicts={pauli_dicts if pauli_dicts else 'pauli_dict'}, name={name_str})

            # 前向传播
            x = tensor.randn([2, 5])
            x.requires_grad = True
            y = layer(x)
            print("输出:", y)

            # 反向传播
            y.backward()
            print("参数梯度:", layer.m_para.grad)
            print("输入梯度:", x.grad)
            """

            return {
                "status": "success",
                "message": f"已生成QuantumLayerAdjoint伴随法梯度计算层代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet QuantumLayerAdjoint API生成的真实代码，使用伴随法计算量子电路参数梯度。"
            }
        elif "pyvqnet.qnn.pq3.template.CSWAPcircuit" in tool_name or "CSWAPcircuit" in tool_name:
            # CSWAPcircuit 受控SWAP线路
            qubits = arguments.get('qubits')

            generated_code = f"""
            from pyvqnet.qnn.pq3 import CSWAPcircuit
            import pyqpanda3.core as pq

            # 创建受控SWAP线路
            # 第一个量子比特为控制比特
            m_qlist = range(3)
            c = CSWAPcircuit({qubits if qubits else '[m_qlist[0], m_qlist[1], m_qlist[2]]'})
            print("CSWAP线路:")
            print(c)
            """

            return {
                "status": "success",
                "message": f"已生成CSWAPcircuit受控SWAP线路代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet CSWAPcircuit API生成的真实代码，创建受控SWAP量子线路。"
            }
        elif "pyvqnet.qnn.pq3.template.Controlled_Hadamard" in tool_name or "Controlled_Hadamard" in tool_name:
            # Controlled_Hadamard 受控Hadamard门
            qubits = arguments.get('qubits')

            generated_code = f"""
            from pyvqnet.qnn.pq3 import Controlled_Hadamard
            import pyqpanda3.core as pq

            # 创建受控Hadamard门
            qubits = {qubits if qubits else 'range(2)'}
            cir = Controlled_Hadamard(qubits)
            print("受控Hadamard线路:")
            print(cir)
            """

            return {
                "status": "success",
                "message": f"已生成Controlled_Hadamard受控Hadamard门代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet Controlled_Hadamard API生成的真实代码，创建受控Hadamard量子门。"
            }
        elif "pyvqnet.qnn.pq3.template.CCZ" in tool_name or "CCZ" in tool_name:
            # CCZ 受控-受控-Z门
            qubits = arguments.get('qubits')

            generated_code = f"""
            from pyvqnet.qnn.pq3 import CCZ
            import pyqpanda3.core as pq

            # 创建CCZ门（受控-受控-Z）
            qubits = {qubits if qubits else 'range(3)'}
            cir = CCZ(qubits)
            print("CCZ线路:")
            print(cir)
            """

            return {
                "status": "success",
                "message": f"已生成CCZ受控-受控-Z门代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet CCZ API生成的真实代码，创建CCZ量子门。"
            }
        elif "pyvqnet.qnn.pq3.ansatz.HardwareEfficientAnsatz" in tool_name:
            # HardwareEfficientAnsatz 硬件高效拟设
            qubits = arguments.get('qubits')
            single_rot_gate_list = arguments.get('single_rot_gate_list')
            entangle_gate = arguments.get('entangle_gate', 'CNOT')
            entangle_rules = arguments.get('entangle_rules', 'linear')
            depth = arguments.get('depth', 1)

            gate_list_str = str(single_rot_gate_list) if single_rot_gate_list else '["rx", "RY", "rz"]'

            generated_code = f"""
            import pyqpanda3.core as pq
            from pyvqnet.tensor import QTensor, tensor
            from pyvqnet.qnn.pq3.ansatz import HardwareEfficientAnsatz

            # 创建硬件高效拟设
            qlist = {qubits if qubits else 'range(4)'}
            c = HardwareEfficientAnsatz(
                qlist,
                {gate_list_str},
                entangle_gate="{entangle_gate}",
                entangle_rules="{entangle_rules}",
                depth={depth}
            )

            # 生成参数
            w = tensor.ones([c.get_para_num()])

            # 创建线路
            cir = c.create_ansatz(w)
            print("HardwareEfficientAnsatz线路:")
            print(cir)
            print(f"参数数量: {{c.get_para_num()}}")
            """

            return {
                "status": "success",
                "message": f"已生成HardwareEfficientAnsatz硬件高效拟设代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet HardwareEfficientAnsatz API生成的真实代码，创建硬件高效变分量子线路。"
            }
        elif "pyvqnet.qnn.pq3.template.BasicEntanglerTemplate" in tool_name or "BasicEntanglerTemplate" in tool_name:
            # BasicEntanglerTemplate 基础纠缠模板
            weights = arguments.get('weights')
            num_qubits = arguments.get('num_qubits', 1)
            rotation = arguments.get('rotation', 'pq.RX')

            generated_code = f"""
            import pyqpanda3.core as pq
            import numpy as np
            from pyvqnet.qnn.pq3 import BasicEntanglerTemplate

            # 设置随机种子
            np.random.seed(42)

            # 创建基础纠缠模板
            num_qubits = {num_qubits}
            shape = [1, num_qubits]
            weights = {weights if weights else 'np.random.random(size=shape)'}

            qubits = range(num_qubits)
            circuit = BasicEntanglerTemplate(weights=weights, num_qubits=num_qubits, rotation={rotation})

            # 创建线路
            result = circuit.compute_circuit()
            cir = circuit.create_circuit(qubits)
            print("BasicEntanglerTemplate线路:")
            circuit.print_circuit(qubits)
            """

            return {
                "status": "success",
                "message": f"已生成BasicEntanglerTemplate基础纠缠模板代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet BasicEntanglerTemplate API生成的真实代码，创建基础纠缠量子线路模板。"
            }
        elif "pyvqnet.qnn.pq3.template.StronglyEntanglingTemplate" in tool_name or "StronglyEntanglingTemplate" in tool_name:
            # StronglyEntanglingTemplate 强纠缠模板
            weights = arguments.get('weights')
            num_qubits = arguments.get('num_qubits', 1)
            ranges = arguments.get('ranges', None)

            generated_code = f"""
            from pyvqnet.qnn.pq3 import StronglyEntanglingTemplate
            import pyqpanda3.core as pq
            from pyvqnet.tensor import *
            import numpy as np

            # 设置随机种子
            np.random.seed(42)

            # 创建强纠缠模板
            num_qubits = {num_qubits}
            shape = [2, num_qubits, 3]
            weights = {weights if weights else 'np.random.random(size=shape)'}

            qubits = range(num_qubits)
            circuit = StronglyEntanglingTemplate(weights, num_qubits=num_qubits{', ranges=' + str(ranges) if ranges else ''})

            # 创建线路
            result = circuit.compute_circuit()
            cir = circuit.create_circuit(qubits)
            print("StronglyEntanglingTemplate线路:")
            circuit.print_circuit(qubits)
            """

            return {
                "status": "success",
                "message": f"已生成StronglyEntanglingTemplate强纠缠模板代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet StronglyEntanglingTemplate API生成的真实代码，创建强纠缠量子线路模板。"
            }
        elif "pyvqnet.qnn.pq3.ComplexEntangelingTemplate" in tool_name or "ComplexEntangelingTemplate" in tool_name:
            # ComplexEntangelingTemplate 复杂纠缠模板
            weights = arguments.get('weights')
            num_qubits = arguments.get('num_qubits')
            depth = arguments.get('depth')

            generated_code = f"""
            from pyvqnet.qnn.pq3 import ComplexEntangelingTemplate
            import pyqpanda3.core as pq
            from pyvqnet.tensor import *

            # 创建复杂纠缠模板
            depth = {depth if depth else 3}
            num_qubits = {num_qubits if num_qubits else 8}
            shape = [depth, num_qubits, 3]
            weights = {weights if weights else 'tensor.randn(shape)'}

            qubits = range(num_qubits)
            circuit = ComplexEntangelingTemplate(weights, num_qubits=num_qubits, depth=depth)

            # 创建线路
            result = circuit.create_circuit(qubits)
            print("ComplexEntangelingTemplate线路:")
            circuit.print_circuit(qubits)
            """

            return {
                "status": "success",
                "message": f"已生成ComplexEntangelingTemplate复杂纠缠模板代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet ComplexEntangelingTemplate API生成的真实代码，创建复杂纠缠量子线路模板。"
            }
        elif "pyvqnet.qnn.pq3.Quantum_Embedding" in tool_name or "Quantum_Embedding" in tool_name:
            # Quantum_Embedding 量子嵌入
            num_repetitions_input = arguments.get('num_repetitions_input')
            depth_input = arguments.get('depth_input')
            num_unitary_layers = arguments.get('num_unitary_layers')
            num_repetitions = arguments.get('num_repetitions')

            generated_code = f"""
            from pyvqnet.qnn.pq3 import QpandaQCircuitVQCLayerLite, Quantum_Embedding
            from pyvqnet.tensor import tensor
            import pyqpanda3.core as pq

            # 设置参数
            depth_input = {depth_input if depth_input else 2}
            num_repetitions = {num_repetitions if num_repetitions else 2}
            num_repetitions_input = {num_repetitions_input if num_repetitions_input else 2}
            num_unitary_layers = {num_unitary_layers if num_unitary_layers else 2}

            # 创建量子机器
            local_machine = pq.CPUQVM()

            # 计算量子比特数
            nq = depth_input * num_repetitions_input
            qubits = range(nq)
            cubits = range(nq)

            # 创建输入数据
            data_in = tensor.ones([12, depth_input])
            data_in.requires_grad = True

            # 创建量子嵌入
            qe = Quantum_Embedding(nq, local_machine, num_repetitions_input,
                                    depth_input, num_unitary_layers, num_repetitions)
            qlayer = QpandaQCircuitVQCLayerLite(qe.compute_circuit, qe.param_num)

            # 前向传播
            y = qlayer.forward(data_in)
            print("输出:", y)

            # 反向传播
            y.backward()
            print("输入梯度:", data_in.grad)
            """

            return {
                "status": "success",
                "message": f"已生成Quantum_Embedding量子嵌入代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet Quantum_Embedding API生成的真实代码，创建量子嵌入线路。"
            }
        elif "pyvqnet.qnn.pq3.template.QuantumPoolingCircuit" in tool_name or "QuantumPoolingCircuit" in tool_name:
            # QuantumPoolingCircuit 量子池化线路
            sources_wires = arguments.get('sources_wires')
            sinks_wires = arguments.get('sinks_wires')
            params = arguments.get('params')

            generated_code = f"""
            from pyvqnet.qnn.pq3.template import QuantumPoolingCircuit
            import pyqpanda3.core as pq
            from pyvqnet import tensor

            # 创建量子池化线路
            qlists = range(4)
            p = {params if params else 'tensor.full([6], 0.35)'}

            cir = QuantumPoolingCircuit(
                {sources_wires if sources_wires else '[0, 1]'},
                {sinks_wires if sinks_wires else '[2, 3]'},
                p,
                qlists
            )
            print("QuantumPoolingCircuit线路:")
            print(cir)
            """

            return {
                "status": "success",
                "message": f"已生成QuantumPoolingCircuit量子池化线路代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet QuantumPoolingCircuit API生成的真实代码，创建量子池化线路。"
            }
        elif "pyvqnet.qnn.pq3.template.FermionicSingleExcitation" in tool_name or "FermionicSingleExcitation" in tool_name:
            # FermionicSingleExcitation 费米子单激发算子
            weight = arguments.get('weight')
            wires = arguments.get('wires')
            qubits = arguments.get('qubits')

            generated_code = f"""
            from pyvqnet.qnn.pq3 import FermionicSingleExcitation, expval
            import pyqpanda3.core as pq

            # 创建费米子单激发算子
            weight = {weight if weight else 0.5}
            qlists = range(3)

            cir = FermionicSingleExcitation(
                weight,
                {wires if wires else '[1, 0, 2]'},
                {qubits if qubits else 'qlists'}
            )
            print("FermionicSingleExcitation线路:")
            print(cir)
            """

            return {
                "status": "success",
                "message": f"已生成FermionicSingleExcitation费米子单激发算子代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet FermionicSingleExcitation API生成的真实代码，创建费米子单激发量子线路。"
            }
        elif "pyvqnet.qnn.pq3.template.FermionicDoubleExcitation" in tool_name or "FermionicDoubleExcitation" in tool_name:
            # FermionicDoubleExcitation 费米子双激发算子
            weight = arguments.get('weight')
            wires1 = arguments.get('wires1')
            wires2 = arguments.get('wires2')
            qubits = arguments.get('qubits')

            generated_code = f"""
            from pyvqnet.qnn.pq3 import FermionicDoubleExcitation, expval
            import pyqpanda3.core as pq

            # 创建费米子双激发算子
            weight = {weight if weight else 1.5}
            qlists = range(5)

            cir = FermionicDoubleExcitation(
                weight,
                wires1={wires1 if wires1 else '[0, 1]'},
                wires2={wires2 if wires2 else '[2, 3, 4]'},
                qubits={qubits if qubits else 'qlists'}
            )
            print("FermionicDoubleExcitation线路:")
            print(cir)
            """

            return {
                "status": "success",
                "message": f"已生成FermionicDoubleExcitation费米子双激发算子代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet FermionicDoubleExcitation API生成的真实代码，创建费米子双激发量子线路。"
            }
        elif "pyvqnet.qnn.pq3.measure.ProbsMeasure" in tool_name or "ProbsMeasure" in tool_name:
            # ProbsMeasure 概率测量
            prog = arguments.get('prog')
            measure_qubits = arguments.get('measure_qubits')
            shots = arguments.get('shots', 1)

            generated_code = f"""
            from pyqpanda3.core import *
            from pyvqnet.qnn.pq3.measure import probs_measure

            # 创建量子线路
            circuit = QCircuit(3)
            circuit << H(0)
            circuit << P(2, 0.2)
            circuit << RX(1, 0.9)
            circuit << RX(0, 0.6)
            circuit << RX(1, 0.3)
            circuit << RY(1, 0.3)
            circuit << RY(2, 2.7)
            circuit << RX(0, 1.5)

            prog = QProg()
            prog.append(circuit)
            prog.append(measure(0, 0))
            prog.append(measure(1, 1))
            prog.append(measure(2, 2))

            # 创建量子虚拟机
            machine = CPUQVM()

            # 执行概率测量
            measure_result = probs_measure(machine, prog, {measure_qubits if measure_qubits else '[2, 0]'}, shots={shots})
            print("概率测量结果:", measure_result)
            """

            return {
                "status": "success",
                "message": f"已生成ProbsMeasure概率测量代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet ProbsMeasure API生成的真实代码，计算量子线路的概率分布。"
            }
        elif "pyvqnet.qnn.pq3.measure.DensityMatrixFromQstate" in tool_name or "DensityMatrixFromQstate" in tool_name:
            # DensityMatrixFromQstate 密度矩阵计算
            state = arguments.get('state')
            indices = arguments.get('indices')

            generated_code = f"""
            from pyvqnet.qnn.pq3.measure import DensityMatrixFromQstate

            # 定义量子态
            qstate = {state if state else '[(0.9306699299765968+0j), (0.18865613455240968+0j), (0.1886561345524097+0j), (0.03824249173404786+0j), -0.048171819846746615j, -0.00976491131165138j, -0.23763904794287155j, -0.048171819846746615j]'}

            # 计算子系统密度矩阵
            dm = DensityMatrixFromQstate(qstate, {indices if indices else '[0, 1]'})
            print("密度矩阵:")
            print(dm)
            """

            return {
                "status": "success",
                "message": f"已生成DensityMatrixFromQstate密度矩阵计算代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet DensityMatrixFromQstate API生成的真实代码，计算子系统的密度矩阵。"
            }
        elif "pyvqnet.qnn.pq3.measure.VN_Entropy" in tool_name or "VN_Entropy" in tool_name:
            # VN_Entropy 冯诺依曼熵
            state = arguments.get('state')
            indices = arguments.get('indices')
            base = arguments.get('base', None)

            base_str = base if base is not None else 'None'

            generated_code = f"""
            from pyvqnet.qnn.pq3.measure import VN_Entropy

            # 定义量子态
            qstate = {state if state else '[(0.9022961387408862 + 0j), -0.06676534788028633j, (0.18290448232350312 + 0j), -0.3293638014158896j, (0.03707657410649268 + 0j), -0.06676534788028635j, (0.18290448232350312 + 0j), -0.013534006039561714j]'}

            # 计算冯诺依曼熵
            entropy = VN_Entropy(qstate, {indices if indices else '[0, 1]'}, base={base_str})
            print("冯诺依曼熵:", entropy)
            """

            return {
                "status": "success",
                "message": f"已生成VN_Entropy冯诺依曼熵计算代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet VN_Entropy API生成的真实代码，计算子系统的冯诺依曼熵。"
            }
        elif "pyvqnet.qnn.pq3.measure.Mutal_Info" in tool_name or "Mutal_Info" in tool_name:
            # Mutal_Info 互信息
            state = arguments.get('state')
            indices0 = arguments.get('indices0')
            indices1 = arguments.get('indices1')
            base = arguments.get('base', None)

            base_str = base if base is not None else 'None'

            generated_code = f"""
            from pyvqnet.qnn.pq3.measure import Mutal_Info

            # 定义量子态
            qstate = {state if state else '[(0.9022961387408862 + 0j), -0.06676534788028633j, (0.18290448232350312 + 0j), -0.3293638014158896j, (0.03707657410649268 + 0j), -0.06676534788028635j, (0.18290448232350312 + 0j), -0.013534006039561714j]'}

            # 计算互信息
            mi = Mutal_Info(qstate, {indices0 if indices0 else '[0]'}, {indices1 if indices1 else '[2]'}, base={base_str})
            print("互信息:", mi)
            """

            return {
                "status": "success",
                "message": f"已生成Mutal_Info互信息计算代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet Mutal_Info API生成的真实代码，计算两个子系统间的互信息。"
            }
        elif "pyvqnet.qnn.pq3.measure.Purity" in tool_name or "Purity" in tool_name:
            # Purity 纯度
            state = arguments.get('state')
            qubits_idx = arguments.get('qubits_idx')

            generated_code = f"""
            from pyvqnet.qnn.pq3.measure import Purity

            # 定义量子态
            qstate = {state if state else '[(0.9022961387408862 + 0j), -0.06676534788028633j, (0.18290448232350312 + 0j), -0.3293638014158896j, (0.03707657410649268 + 0j), -0.06676534788028635j, (0.18290448232350312 + 0j), -0.013534006039561714j]'}

            # 计算纯度
            purity = Purity(qstate, {qubits_idx if qubits_idx else '[0, 1]'})
            print("纯度:", purity)
            """

            return {
                "status": "success",
                "message": f"已生成Purity纯度计算代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet Purity API生成的真实代码，计算子系统的纯度。"
            }
        # =====================  NN Module Missing Implementations =====================
        elif "pyvqnet.nn.BatchNorm1d" in tool_name:
            # BatchNorm1d 一维批量归一化层
            num_features = arguments.get('num_features')
            eps = arguments.get('eps', 1e-05)
            momentum = arguments.get('momentum', 0.1)
            affine = arguments.get('affine', True)
            track_running_stats = arguments.get('track_running_stats', True)

            generated_code = f"""
            from pyvqnet.nn import BatchNorm1d
            from pyvqnet.tensor import tensor

            # 创建BatchNorm1d层
            bn = BatchNorm1d(
                num_features={num_features if num_features else 3},
                eps={eps},
                momentum={momentum},
                affine={affine},
                track_running_stats={track_running_stats}
            )

            # 测试输入
            x = tensor.randn([2, 3, 10])
            y = bn(x)
            print("输出形状:", y.shape)
            print("输出:", y)
            """

            return {
                "status": "success",
                "message": f"已生成BatchNorm1d一维批量归一化层代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet BatchNorm1d API生成的真实代码，创建一维批量归一化层。"
            }
        elif "pyvqnet.nn.HardSigmoid" in tool_name or "HardSigmoid" in tool_name:
            # HardSigmoid 激活函数
            generated_code = f"""
            from pyvqnet.nn import HardSigmoid
            from pyvqnet.tensor import tensor

            # 创建HardSigmoid激活函数
            act = HardSigmoid()

            # 测试输入
            x = tensor.randn([2, 3])
            y = act(x)
            print("输入:", x)
            print("输出:", y)
            """

            return {
                "status": "success",
                "message": f"已生成HardSigmoid激活函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet HardSigmoid API生成的真实代码，HardSigmoid是Sigmoid的分段线性近似，计算更快。"
            }
        elif "pyvqnet.nn.LeakyReLu" in tool_name:
            # LeakyReLu 激活函数
            alpha = arguments.get('alpha', 0.01)

            generated_code = f"""
            from pyvqnet.nn import LeakyReLu
            from pyvqnet.tensor import tensor

            # 创建LeakyReLu激活函数
            act = LeakyReLu(alpha={alpha})

            # 测试输入
            x = tensor.randn([2, 3])
            y = act(x)
            print("输入:", x)
            print("输出:", y)
            """

            return {
                "status": "success",
                "message": f"已生成LeakyReLu激活函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet LeakyReLu API生成的真实代码，带泄露的ReLU激活函数。"
            }
        elif "pyvqnet.nn.Softplus" in tool_name or "Softplus" in tool_name:
            # Softplus 激活函数
            generated_code = f"""
            from pyvqnet.nn import Softplus
            from pyvqnet.tensor import tensor

            # 创建Softplus激活函数
            act = Softplus()

            # 测试输入
            x = tensor.randn([2, 3])
            y = act(x)
            print("输入:", x)
            print("输出:", y)
            """

            return {
                "status": "success",
                "message": f"已生成Softplus激活函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet Softplus API生成的真实代码，Softplus是平滑的ReLU近似。"
            }
        elif "pyvqnet.nn.Softsign" in tool_name or "Softsign" in tool_name:
            # Softsign 激活函数
            generated_code = f"""
            from pyvqnet.nn import Softsign
            from pyvqnet.tensor import tensor

            # 创建Softsign激活函数
            act = Softsign()

            # 测试输入
            x = tensor.randn([2, 3])
            y = act(x)
            print("输入:", x)
            print("输出:", y)
            """

            return {
                "status": "success",
                "message": f"已生成Softsign激活函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet Softsign API生成的真实代码，Softsign是x / (1 + |x|)的激活函数。"
            }
        elif "pyvqnet.nn.Tanh" in tool_name or "Tanh" in tool_name:
            # Tanh 激活函数
            generated_code = f"""
            from pyvqnet.nn import Tanh
            from pyvqnet.tensor import tensor

            # 创建Tanh激活函数
            act = Tanh()

            # 测试输入
            x = tensor.randn([2, 3])
            y = act(x)
            print("输入:", x)
            print("输出:", y)
            """

            return {
                "status": "success",
                "message": f"已生成Tanh激活函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet Tanh API生成的真实代码，双曲正切激活函数。"
            }
        elif "pyvqnet.nn.dropout.DropPath" in tool_name or "DropPath" in tool_name:
            # DropPath 随机丢弃路径
            drop_prob = arguments.get('drop_prob', 0.0)

            generated_code = f"""
            from pyvqnet.nn.dropout import DropPath
            from pyvqnet.tensor import tensor

            # 创建DropPath层
            dp = DropPath(drop_prob={drop_prob})

            # 测试输入
            x = tensor.randn([2, 3, 4, 4])
            y = dp(x)
            print("输入形状:", x.shape)
            print("输出形状:", y.shape)
            """

            return {
                "status": "success",
                "message": f"已生成DropPath随机丢弃路径代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet DropPath API生成的真实代码，随机丢弃整个样本路径用于正则化。"
            }
        elif "pyvqnet.nn.group_norm.GroupNorm" in tool_name:
            # GroupNorm 组归一化层
            num_groups = arguments.get('num_groups')
            num_channels = arguments.get('num_channels')
            eps = arguments.get('eps', 1e-05)
            affine = arguments.get('affine', True)

            generated_code = f"""
            from pyvqnet.nn.group_norm import GroupNorm
            from pyvqnet.tensor import tensor

            # 创建GroupNorm层
            gn = GroupNorm(
                num_groups={num_groups if num_groups else 2},
                num_channels={num_channels if num_channels else 4},
                eps={eps},
                affine={affine}
            )

            # 测试输入
            x = tensor.randn([2, 4, 10, 10])
            y = gn(x)
            print("输出形状:", y.shape)
            print("输出:", y)
            """

            return {
                "status": "success",
                "message": f"已生成GroupNorm组归一化层代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet GroupNorm API生成的真实代码，创建组归一化层。"
            }
        elif "pyvqnet.nn.layer_norm.LayerNorm1d" in tool_name:
            # LayerNorm1d 一维层归一化
            normalized_shape = arguments.get('normalized_shape')
            eps = arguments.get('eps', 1e-05)
            elementwise_affine = arguments.get('elementwise_affine', True)

            generated_code = f"""
            from pyvqnet.nn.layer_norm import LayerNorm1d
            from pyvqnet.tensor import tensor

            # 创建LayerNorm1d层
            ln = LayerNorm1d(
                normalized_shape={normalized_shape if normalized_shape else 10},
                eps={eps},
                elementwise_affine={elementwise_affine}
            )

            # 测试输入
            x = tensor.randn([2, 10])
            y = ln(x)
            print("输出形状:", y.shape)
            print("输出:", y)
            """

            return {
                "status": "success",
                "message": f"已生成LayerNorm1d一维层归一化代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet LayerNorm1d API生成的真实代码，创建一维层归一化层。"
            }
        elif "pyvqnet.nn.layer_norm.LayerNorm2d" in tool_name:
            # LayerNorm2d 二维层归一化
            normalized_shape = arguments.get('normalized_shape')
            eps = arguments.get('eps', 1e-05)
            elementwise_affine = arguments.get('elementwise_affine', True)

            generated_code = f"""
            from pyvqnet.nn.layer_norm import LayerNorm2d
            from pyvqnet.tensor import tensor

            # 创建LayerNorm2d层
            ln = LayerNorm2d(
                normalized_shape={normalized_shape if normalized_shape else 3},
                eps={eps},
                elementwise_affine={elementwise_affine}
            )

            # 测试输入
            x = tensor.randn([2, 3, 10, 10])
            y = ln(x)
            print("输出形状:", y.shape)
            print("输出:", y)
            """

            return {
                "status": "success",
                "message": f"已生成LayerNorm2d二维层归一化代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet LayerNorm2d API生成的真实代码，创建二维层归一化层。"
            }
        elif "pyvqnet.nn.layer_norm.LayerNormNd" in tool_name:
            # LayerNormNd N维层归一化
            normalized_shape = arguments.get('normalized_shape')
            eps = arguments.get('eps', 1e-05)
            elementwise_affine = arguments.get('elementwise_affine', True)

            generated_code = f"""
            from pyvqnet.nn.layer_norm import LayerNormNd
            from pyvqnet.tensor import tensor

            # 创建LayerNormNd层
            ln = LayerNormNd(
                normalized_shape={normalized_shape if normalized_shape else 3},
                eps={eps},
                elementwise_affine={elementwise_affine}
            )

            # 测试输入
            x = tensor.randn([2, 3, 10, 10])
            y = ln(x)
            print("输出形状:", y.shape)
            print("输出:", y)
            """

            return {
                "status": "success",
                "message": f"已生成LayerNormNd N维层归一化代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet LayerNormNd API生成的真实代码，创建N维层归一化层。"
            }
        elif "pyvqnet.nn.gru.GRU" in tool_name or "GRU" in tool_name:
            # GRU 门控循环单元
            input_size = arguments.get('input_size')
            hidden_size = arguments.get('hidden_size')
            num_layers = arguments.get('num_layers', 1)
            bias = arguments.get('bias', True)
            batch_first = arguments.get('batch_first', True)
            dropout = arguments.get('dropout', 0.0)
            bidirectional = arguments.get('bidirectional', False)

            generated_code = f"""
            from pyvqnet.nn.gru import GRU
            from pyvqnet.tensor import tensor

            # 创建GRU层
            gru = GRU(
                input_size={input_size if input_size else 10},
                hidden_size={hidden_size if hidden_size else 20},
                num_layers={num_layers},
                bias={bias},
                batch_first={batch_first},
                dropout={dropout},
                bidirectional={bidirectional}
            )

            # 测试输入
            x = tensor.randn([32, 10, 10])  # [batch, seq_len, input_size]
            output, hidden = gru(x)
            print("输出形状:", output.shape)
            print("隐藏状态形状:", hidden.shape)
            """

            return {
                "status": "success",
                "message": f"已生成GRU门控循环单元代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet GRU API生成的真实代码，创建门控循环单元层。"
            }
        elif "pyvqnet.nn.lstm.LSTM" in tool_name or "LSTM" in tool_name:
            # LSTM 长短期记忆网络
            input_size = arguments.get('input_size')
            hidden_size = arguments.get('hidden_size')
            num_layers = arguments.get('num_layers', 1)
            bias = arguments.get('bias', True)
            batch_first = arguments.get('batch_first', True)
            dropout = arguments.get('dropout', 0.0)
            bidirectional = arguments.get('bidirectional', False)

            generated_code = f"""
            from pyvqnet.nn.lstm import LSTM
            from pyvqnet.tensor import tensor

            # 创建LSTM层
            lstm = LSTM(
                input_size={input_size if input_size else 10},
                hidden_size={hidden_size if hidden_size else 20},
                num_layers={num_layers},
                bias={bias},
                batch_first={batch_first},
                dropout={dropout},
                bidirectional={bidirectional}
            )

            # 测试输入
            x = tensor.randn([32, 10, 10])  # [batch, seq_len, input_size]
            output, (hidden, cell) = lstm(x)
            print("输出形状:", output.shape)
            print("隐藏状态形状:", hidden.shape)
            print("细胞状态形状:", cell.shape)
            """

            return {
                "status": "success",
                "message": f"已生成LSTM长短期记忆网络代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet LSTM API生成的真实代码，创建长短期记忆网络层。"
            }
        elif "pyvqnet.nn.rnn.RNN" in tool_name or "RNN" in tool_name:
            # RNN 循环神经网络
            input_size = arguments.get('input_size')
            hidden_size = arguments.get('hidden_size')
            num_layers = arguments.get('num_layers', 1)
            nonlinearity = arguments.get('nonlinearity', 'tanh')
            bias = arguments.get('bias', True)
            batch_first = arguments.get('batch_first', True)
            dropout = arguments.get('dropout', 0.0)
            bidirectional = arguments.get('bidirectional', False)

            generated_code = f"""
            from pyvqnet.nn.rnn import RNN
            from pyvqnet.tensor import tensor

            # 创建RNN层
            rnn = RNN(
                input_size={input_size if input_size else 10},
                hidden_size={hidden_size if hidden_size else 20},
                num_layers={num_layers},
                nonlinearity="{nonlinearity}",
                bias={bias},
                batch_first={batch_first},
                dropout={dropout},
                bidirectional={bidirectional}
            )

            # 测试输入
            x = tensor.randn([32, 10, 10])  # [batch, seq_len, input_size]
            output, hidden = rnn(x)
            print("输出形状:", output.shape)
            print("隐藏状态形状:", hidden.shape)
            """

            return {
                "status": "success",
                "message": f"已生成RNN循环神经网络代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet RNN API生成的真实代码，创建循环神经网络层。"
            }
        elif "pyvqnet.nn.Interpolate" in tool_name or "Interpolate" in tool_name:
            # Interpolate 上采样/下采样
            size = arguments.get('size')
            scale_factor = arguments.get('scale_factor')
            mode = arguments.get('mode', 'nearest')
            align_corners = arguments.get('align_corners', None)

            size_str = str(size) if size is not None else 'None'
            scale_str = str(scale_factor) if scale_factor is not None else 'None'
            align_str = str(align_corners) if align_corners is not None else 'None'

            generated_code = f"""
            from pyvqnet.nn import Interpolate
            from pyvqnet.tensor import tensor

            # 创建Interpolate层
            interp = Interpolate(
                size={size_str},
                scale_factor={scale_str},
                mode="{mode}",
                align_corners={align_str}
            )

            # 测试输入
            x = tensor.randn([2, 3, 16, 16])
            y = interp(x)
            print("输入形状:", x.shape)
            print("输出形状:", y.shape)
            """

            return {
                "status": "success",
                "message": f"已生成Interpolate上采样/下采样代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet Interpolate API生成的真实代码，创建上采样或下采样层。"
            }
        elif "pyvqnet.nn.SoftmaxCrossEntropy" in tool_name or "SoftmaxCrossEntropy" in tool_name:
            # SoftmaxCrossEntropy 损失函数
            generated_code = f"""
            from pyvqnet.nn import SoftmaxCrossEntropy
            from pyvqnet.tensor import tensor

            # 创建损失函数
            loss_fn = SoftmaxCrossEntropy()

            # 测试输入
            logits = tensor.randn([2, 10])  # 预测值
            labels = tensor.tensor([3, 7])   # 标签
            loss = loss_fn(logits, labels)
            print("损失值:", loss)
            """

            return {
                "status": "success",
                "message": f"已生成SoftmaxCrossEntropy损失函数代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet SoftmaxCrossEntropy API生成的真实代码，Softmax和交叉熵组合损失函数。"
            }
        # =====================  Optim Module Missing Implementations =====================
        elif "pyvqnet.optim.adadelta.Adadelta" in tool_name or "Adadelta" in tool_name:
            # Adadelta 优化器
            lr = arguments.get('lr', 1.0)
            rho = arguments.get('rho', 0.9)
            eps = arguments.get('eps', 1e-06)
            weight_decay = arguments.get('weight_decay', 0.0)

            generated_code = f"""
            from pyvqnet.optim.adadelta import Adadelta
            from pyvqnet.nn import Linear
            from pyvqnet.tensor import tensor

            # 创建模型和优化器
            model = Linear(10, 2)
            optimizer = Adadelta(
                model.parameters(),
                lr={lr},
                rho={rho},
                eps={eps},
                weight_decay={weight_decay}
            )

            # 训练步骤示例
            x = tensor.randn([32, 10])
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print("优化器已创建并执行一步更新")
            """

            return {
                "status": "success",
                "message": f"已生成Adadelta优化器代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet Adadelta API生成的真实代码，创建Adadelta优化器。"
            }
        elif "pyvqnet.optim.adagrad.Adagrad" in tool_name or "Adagrad" in tool_name:
            # Adagrad 优化器
            lr = arguments.get('lr', 0.01)
            lr_decay = arguments.get('lr_decay', 0.0)
            weight_decay = arguments.get('weight_decay', 0.0)
            initial_accumulator_value = arguments.get('initial_accumulator_value', 0.0)
            eps = arguments.get('eps', 1e-10)

            generated_code = f"""
            from pyvqnet.optim.adagrad import Adagrad
            from pyvqnet.nn import Linear
            from pyvqnet.tensor import tensor

            # 创建模型和优化器
            model = Linear(10, 2)
            optimizer = Adagrad(
                model.parameters(),
                lr={lr},
                lr_decay={lr_decay},
                weight_decay={weight_decay},
                initial_accumulator_value={initial_accumulator_value},
                eps={eps}
            )

            # 训练步骤示例
            x = tensor.randn([32, 10])
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print("优化器已创建并执行一步更新")
            """

            return {
                "status": "success",
                "message": f"已生成Adagrad优化器代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet Adagrad API生成的真实代码，创建Adagrad优化器。"
            }
        elif "pyvqnet.optim.adam.AdamW" in tool_name or "AdamW" in tool_name:
            # AdamW 优化器
            lr = arguments.get('lr', 0.001)
            betas = arguments.get('betas', (0.9, 0.999))
            eps = arguments.get('eps', 1e-08)
            weight_decay = arguments.get('weight_decay', 0.01)
            amsgrad = arguments.get('amsgrad', False)

            betas_str = str(betas) if betas else '(0.9, 0.999)'

            generated_code = f"""
            from pyvqnet.optim.adam import AdamW
            from pyvqnet.nn import Linear
            from pyvqnet.tensor import tensor

            # 创建模型和优化器
            model = Linear(10, 2)
            optimizer = AdamW(
                model.parameters(),
                lr={lr},
                betas={betas_str},
                eps={eps},
                weight_decay={weight_decay},
                amsgrad={amsgrad}
            )

            # 训练步骤示例
            x = tensor.randn([32, 10])
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print("优化器已创建并执行一步更新")
            """

            return {
                "status": "success",
                "message": f"已生成AdamW优化器代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet AdamW API生成的真实代码，创建带权重衰减的Adam优化器。"
            }
        elif "pyvqnet.optim.adamax.Adamax" in tool_name or "Adamax" in tool_name:
            # Adamax 优化器
            lr = arguments.get('lr', 0.002)
            betas = arguments.get('betas', (0.9, 0.999))
            eps = arguments.get('eps', 1e-08)
            weight_decay = arguments.get('weight_decay', 0.0)

            betas_str = str(betas) if betas else '(0.9, 0.999)'

            generated_code = f"""
            from pyvqnet.optim.adamax import Adamax
            from pyvqnet.nn import Linear
            from pyvqnet.tensor import tensor

            # 创建模型和优化器
            model = Linear(10, 2)
            optimizer = Adamax(
                model.parameters(),
                lr={lr},
                betas={betas_str},
                eps={eps},
                weight_decay={weight_decay}
            )

            # 训练步骤示例
            x = tensor.randn([32, 10])
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print("优化器已创建并执行一步更新")
            """

            return {
                "status": "success",
                "message": f"已生成Adamax优化器代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet Adamax API生成的真实代码，创建Adamax优化器。"
            }
        elif "pyvqnet.optim.rmsprop.RMSProp" in tool_name or "RMSProp" in tool_name:
            # RMSProp 优化器
            lr = arguments.get('lr', 0.01)
            alpha = arguments.get('alpha', 0.99)
            eps = arguments.get('eps', 1e-08)
            weight_decay = arguments.get('weight_decay', 0.0)
            momentum = arguments.get('momentum', 0.0)
            centered = arguments.get('centered', False)

            generated_code = f"""
            from pyvqnet.optim.rmsprop import RMSProp
            from pyvqnet.nn import Linear
            from pyvqnet.tensor import tensor

            # 创建模型和优化器
            model = Linear(10, 2)
            optimizer = RMSProp(
                model.parameters(),
                lr={lr},
                alpha={alpha},
                eps={eps},
                weight_decay={weight_decay},
                momentum={momentum},
                centered={centered}
            )

            # 训练步骤示例
            x = tensor.randn([32, 10])
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print("优化器已创建并执行一步更新")
            """

            return {
                "status": "success",
                "message": f"已生成RMSProp优化器代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet RMSProp API生成的真实代码，创建RMSProp优化器。"
            }
        # =====================  VQC Module Missing Implementations =====================
        elif "pyvqnet.qnn.vqc.VQC_RotCircuit" in tool_name or "VQC_RotCircuit" in tool_name:
            # VQC_RotCircuit 旋转电路模板
            wires = arguments.get('wires')
            params = arguments.get('params')

            generated_code = f"""
            from pyvqnet.qnn.vqc import VQC_RotCircuit, QMachine
            from pyvqnet.tensor import QTensor

            # 创建量子虚拟机
            qm = QMachine(3)
            qm.reset_states(3)

            # 创建旋转电路
            wires = {wires if wires else '[0, 1, 2]'}
            params = {params if params else 'QTensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])'}
            VQC_RotCircuit(q_machine=qm, wires=wires, params=params)

            print("VQC_RotCircuit已应用到量子线路")
            """

            return {
                "status": "success",
                "message": f"已生成VQC_RotCircuit旋转电路代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet VQC_RotCircuit API生成的真实代码，创建VQC旋转电路模板。"
            }
        elif "pyvqnet.qnn.vqc.VQC_VarMeasure" in tool_name or "VQC_VarMeasure" in tool_name:
            # VQC_VarMeasure 变分测量
            wires = arguments.get('wires')
            observable = arguments.get('observable')

            generated_code = f"""
            from pyvqnet.qnn.vqc import VQC_VarMeasure, QMachine, PauliZ
            from pyvqnet.tensor import QTensor

            # 创建量子虚拟机
            qm = QMachine(2)
            qm.reset_states(2)

            # 创建量子线路
            from pyvqnet.qnn.vqc import hadamard
            hadamard(q_machine=qm, wires=0)
            hadamard(q_machine=qm, wires=1)

            # 执行变分测量
            measure = VQC_VarMeasure(
                q_machine=qm,
                wires={wires if wires else '[0, 1]'},
                observable={observable if observable else '[PauliZ(0), PauliZ(1)]'}
            )
            print("测量结果:", measure)
            """

            return {
                "status": "success",
                "message": f"已生成VQC_VarMeasure变分测量代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet VQC_VarMeasure API生成的真实代码，执行变分量子测量。"
            }
        elif "pyvqnet.qnn.vqc.X1" in tool_name or "X1" in tool_name:
            # X1 门 (X的平方根)
            wires = arguments.get('wires')

            generated_code = f"""
            from pyvqnet.qnn.vqc import X1, QMachine, MeasureAll
            from pyvqnet.tensor import QTensor

            # 创建量子虚拟机
            qm = QMachine(1)
            qm.reset_states(1)

            # 应用X1门
            X1(q_machine=qm, wires={wires if wires else 0})

            # 测量
            result = MeasureAll(q_machine=qm, shots=1024)
            print("X1门应用后测量结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成X1门代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet X1 API生成的真实代码，X1门是Pauli X门的平方根。"
            }
        elif "pyvqnet.qnn.vqc.x1" in tool_name or "x1" in tool_name:
            # x1 函数式门
            wires = arguments.get('wires')

            generated_code = f"""
            from pyvqnet.qnn.vqc import x1, QMachine, MeasureAll
            from pyvqnet.tensor import QTensor

            # 创建量子虚拟机
            qm = QMachine(1)
            qm.reset_states(1)

            # 应用x1门（函数式）
            x1(q_machine=qm, wires={wires if wires else 0})

            # 测量
            result = MeasureAll(q_machine=qm, shots=1024)
            print("x1门应用后测量结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成x1函数式门代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet x1 API生成的真实代码，函数式X1门。"
            }
        elif "pyvqnet.qnn.pq3.measure.QuantumMeasure" in tool_name or "QuantumMeasure" in tool_name:
            # QuantumMeasure 量子测量
            measure_qubits = arguments.get('measure_qubits')
            shots = arguments.get('shots', 1000)

            generated_code = f"""
            from pyqpanda3.core import *
            from pyvqnet.qnn.pq3.measure import QuantumMeasure

            # 创建量子线路
            circuit = QCircuit(3)
            circuit << H(0)
            circuit << H(1)
            circuit << CNOT(0, 2)

            prog = QProg()
            prog.append(circuit)

            # 创建量子虚拟机
            machine = CPUQVM()

            # 执行量子测量
            measure_result = QuantumMeasure(
                machine,
                prog,
                measure_qubits={measure_qubits if measure_qubits else '[2, 0]'},
                shots={shots}
            )
            print("量子测量结果:", measure_result)
            """

            return {
                "status": "success",
                "message": f"已生成QuantumMeasure量子测量代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet QuantumMeasure API生成的真实代码，执行量子线路测量。"
            }
        # =====================  QTensor Module Missing Implementations =====================
        elif "pyvqnet.tensor.bitwise_and" in tool_name or "bitwise_and" in tool_name:
            # bitwise_and 按位与
            t1 = arguments.get('t1')
            t2 = arguments.get('t2')

            generated_code = f"""
            from pyvqnet.tensor import bitwise_and, tensor

            # 创建输入张量
            t1 = {t1 if t1 else 'tensor([1, 2, 3, 4], dtype="int32")'}
            t2 = {t2 if t2 else 'tensor([3, 1, 4, 2], dtype="int32")'}

            # 按位与操作
            result = bitwise_and(t1, t2)
            print("输入1:", t1)
            print("输入2:", t2)
            print("按位与结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成bitwise_and按位与代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet bitwise_and API生成的真实代码，执行按位与操作。"
            }
        elif "pyvqnet.tensor.broadcast" in tool_name or "broadcast" in tool_name:
            # broadcast 广播
            t = arguments.get('t')
            shape = arguments.get('shape')

            generated_code = f"""
            from pyvqnet.tensor import broadcast, tensor

            # 创建输入张量
            t = {t if t else 'tensor([1, 2, 3])'}
            shape = {shape if shape else '(2, 3)'}

            # 广播操作
            result = broadcast(t, shape)
            print("输入:", t)
            print("目标形状:", shape)
            print("广播结果:", result)
            print("结果形状:", result.shape)
            """

            return {
                "status": "success",
                "message": f"已生成broadcast广播代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet broadcast API生成的真实代码，执行张量广播。"
            }
        elif "pyvqnet.tensor.broadcast_to" in tool_name or "broadcast_to" in tool_name:
            # broadcast_to 广播到指定形状
            t = arguments.get('t')
            shape = arguments.get('shape')

            generated_code = f"""
            from pyvqnet.tensor import broadcast_to, tensor

            # 创建输入张量
            t = {t if t else 'tensor([1, 2, 3])'}
            shape = {shape if shape else '(2, 3)'}

            # 广播操作
            result = broadcast_to(t, shape)
            print("输入:", t)
            print("目标形状:", shape)
            print("广播结果:", result)
            print("结果形状:", result.shape)
            """

            return {
                "status": "success",
                "message": f"已生成broadcast_to广播代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet broadcast_to API生成的真实代码，广播张量到指定形状。"
            }
        elif "pyvqnet.tensor.clip" in tool_name or "clip" in tool_name:
            # clip 截断
            t = arguments.get('t')
            min_val = arguments.get('min_val')
            max_val = arguments.get('max_val')

            generated_code = f"""
            from pyvqnet.tensor import clip, tensor

            # 创建输入张量
            t = {t if t else 'tensor([-1, 2, -3, 4, -5])'}
            min_val = {min_val if min_val is not None else '-2'}
            max_val = {max_val if max_val is not None else '3'}

            # 截断操作
            result = clip(t, min_val, max_val)
            print("输入:", t)
            print(f"截断到[{min_val}, {max_val}]")
            print("结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成clip截断代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet clip API生成的真实代码，截断张量值到指定范围。"
            }
        elif "pyvqnet.tensor.equal" in tool_name or "equal" in tool_name:
            # equal 元素相等比较
            t1 = arguments.get('t1')
            t2 = arguments.get('t2')

            generated_code = f"""
            from pyvqnet.tensor import equal, tensor

            # 创建输入张量
            t1 = {t1 if t1 else 'tensor([1, 2, 3, 4])'}
            t2 = {t2 if t2 else 'tensor([1, 3, 3, 5])'}

            # 相等比较
            result = equal(t1, t2)
            print("输入1:", t1)
            print("输入2:", t2)
            print("相等比较结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成equal元素相等比较代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet equal API生成的真实代码，逐元素比较张量是否相等。"
            }
        elif "pyvqnet.tensor.flip" in tool_name or "flip" in tool_name:
            # flip 翻转
            t = arguments.get('t')
            dims = arguments.get('dims')

            dims_str = str(dims) if dims is not None else '0'

            generated_code = f"""
            from pyvqnet.tensor import flip, tensor

            # 创建输入张量
            t = {t if t else 'tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])'}
            dims = {dims_str}

            # 翻转操作
            result = flip(t, dims)
            print("输入:", t)
            print(f"翻转维度: {dims}")
            print("翻转结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成flip翻转代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet flip API生成的真实代码，沿指定维度翻转张量。"
            }
        elif "pyvqnet.tensor.full_like" in tool_name or "full_like" in tool_name:
            # full_like 生成同形状张量
            t = arguments.get('t')
            fill_value = arguments.get('fill_value')
            dtype = arguments.get('dtype', None)
            device = arguments.get('device', None)

            dtype_str = dtype if dtype is not None else 'None'
            device_str = device if device is not None else 'None'

            generated_code = f"""
            from pyvqnet.tensor import full_like, tensor

            # 创建输入张量
            t = {t if t else 'tensor([[1, 2, 3], [4, 5, 6]])'}
            fill_value = {fill_value if fill_value is not None else 5}

            # 生成同形状张量
            result = full_like(t, fill_value, dtype={dtype_str}, device={device_str})
            print("输入张量:", t)
            print("输入形状:", t.shape)
            print("填充值:", fill_value)
            print("结果:", result)
            print("结果形状:", result.shape)
            """

            return {
                "status": "success",
                "message": f"已生成full_like生成同形状张量代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet full_like API生成的真实代码，生成与输入同形状的填充张量。"
            }
        elif "pyvqnet.tensor.gather" in tool_name or "gather" in tool_name:
            # gather 收集
            t = arguments.get('t')
            indices = arguments.get('indices')
            dim = arguments.get('dim', 0)

            generated_code = f"""
            from pyvqnet.tensor import gather, tensor

            # 创建输入张量
            t = {t if t else 'tensor([[1, 2], [3, 4], [5, 6]])'}
            indices = {indices if indices else 'tensor([0, 2])'}
            dim = {dim}

            # 收集操作
            result = gather(t, indices, dim)
            print("输入张量:", t)
            print("索引:", indices)
            print(f"维度: {dim}")
            print("收集结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成gather收集代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet gather API生成的真实代码，沿指定维度收集元素。"
            }
        elif "pyvqnet.tensor.greater_equal" in tool_name or "greater_equal" in tool_name:
            # greater_equal 大于等于比较
            t1 = arguments.get('t1')
            t2 = arguments.get('t2')

            generated_code = f"""
            from pyvqnet.tensor import greater_equal, tensor

            # 创建输入张量
            t1 = {t1 if t1 else 'tensor([1, 3, 2, 4])'}
            t2 = {t2 if t2 else 'tensor([2, 2, 2, 3])'}

            # 大于等于比较
            result = greater_equal(t1, t2)
            print("输入1:", t1)
            print("输入2:", t2)
            print("大于等于比较结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成greater_equal大于等于比较代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet greater_equal API生成的真实代码，逐元素比较大于等于。"
            }
        elif "pyvqnet.tensor.isfinite" in tool_name or "isfinite" in tool_name:
            # isfinite 检查有限值
            t = arguments.get('t')

            generated_code = f"""
            from pyvqnet.tensor import isfinite, tensor
            import numpy as np

            # 创建输入张量（包含inf和nan）
            t = {t if t else 'tensor([1.0, np.inf, np.nan, -np.inf, 5.0])'}

            # 检查有限值
            result = isfinite(t)
            print("输入:", t)
            print("是否有限:", result)
            """

            return {
                "status": "success",
                "message": f"已生成isfinite检查有限值代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet isfinite API生成的真实代码，逐元素检查是否为有限值。"
            }
        elif "pyvqnet.tensor.isinf" in tool_name or "isinf" in tool_name:
            # isinf 检查无穷值
            t = arguments.get('t')

            generated_code = f"""
            from pyvqnet.tensor import isinf, tensor
            import numpy as np

            # 创建输入张量（包含inf和nan）
            t = {t if t else 'tensor([1.0, np.inf, np.nan, -np.inf, 5.0])'}

            # 检查无穷值
            result = isinf(t)
            print("输入:", t)
            print("是否无穷:", result)
            """

            return {
                "status": "success",
                "message": f"已生成isinf检查无穷值代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet isinf API生成的真实代码，逐元素检查是否为无穷值。"
            }
        elif "pyvqnet.tensor.isnan" in tool_name or "isnan" in tool_name:
            # isnan 检查NaN
            t = arguments.get('t')

            generated_code = f"""
            from pyvqnet.tensor import isnan, tensor
            import numpy as np

            # 创建输入张量（包含inf和nan）
            t = {t if t else 'tensor([1.0, np.inf, np.nan, -np.inf, 5.0])'}

            # 检查NaN
            result = isnan(t)
            print("输入:", t)
            print("是否NaN:", result)
            """

            return {
                "status": "success",
                "message": f"已生成isnan检查NaN代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet isnan API生成的真实代码，逐元素检查是否为NaN。"
            }
        elif "pyvqnet.tensor.isneginf" in tool_name or "isneginf" in tool_name:
            # isneginf 检查负无穷
            t = arguments.get('t')

            generated_code = f"""
            from pyvqnet.tensor import isneginf, tensor
            import numpy as np

            # 创建输入张量（包含inf和nan）
            t = {t if t else 'tensor([1.0, np.inf, np.nan, -np.inf, 5.0])'}

            # 检查负无穷
            result = isneginf(t)
            print("输入:", t)
            print("是否负无穷:", result)
            """

            return {
                "status": "success",
                "message": f"已生成isneginf检查负无穷代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet isneginf API生成的真实代码，逐元素检查是否为负无穷。"
            }
        elif "pyvqnet.tensor.isposinf" in tool_name or "isposinf" in tool_name:
            # isposinf 检查正无穷
            t = arguments.get('t')

            generated_code = f"""
            from pyvqnet.tensor import isposinf, tensor
            import numpy as np

            # 创建输入张量（包含inf和nan）
            t = {t if t else 'tensor([1.0, np.inf, np.nan, -np.inf, 5.0])'}

            # 检查正无穷
            result = isposinf(t)
            print("输入:", t)
            print("是否正无穷:", result)
            """

            return {
                "status": "success",
                "message": f"已生成isposinf检查正无穷代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet isposinf API生成的真实代码，逐元素检查是否为正无穷。"
            }
        elif "pyvqnet.tensor.less" in tool_name or "less" in tool_name:
            # less 小于比较
            t1 = arguments.get('t1')
            t2 = arguments.get('t2')

            generated_code = f"""
            from pyvqnet.tensor import less, tensor

            # 创建输入张量
            t1 = {t1 if t1 else 'tensor([1, 3, 2, 4])'}
            t2 = {t2 if t2 else 'tensor([2, 2, 2, 3])'}

            # 小于比较
            result = less(t1, t2)
            print("输入1:", t1)
            print("输入2:", t2)
            print("小于比较结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成less小于比较代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet less API生成的真实代码，逐元素比较小于。"
            }
        elif "pyvqnet.tensor.less_equal" in tool_name or "less_equal" in tool_name:
            # less_equal 小于等于比较
            t1 = arguments.get('t1')
            t2 = arguments.get('t2')

            generated_code = f"""
            from pyvqnet.tensor import less_equal, tensor

            # 创建输入张量
            t1 = {t1 if t1 else 'tensor([1, 3, 2, 4])'}
            t2 = {t2 if t2 else 'tensor([2, 2, 2, 3])'}

            # 小于等于比较
            result = less_equal(t1, t2)
            print("输入1:", t1)
            print("输入2:", t2)
            print("小于等于比较结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成less_equal小于等于比较代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet less_equal API生成的真实代码，逐元素比较小于等于。"
            }
        elif "pyvqnet.tensor.linspace" in tool_name or "linspace" in tool_name:
            # linspace 生成等间距序列
            start = arguments.get('start')
            end = arguments.get('end')
            steps = arguments.get('steps')
            dtype = arguments.get('dtype', None)
            device = arguments.get('device', None)

            dtype_str = dtype if dtype is not None else 'None'
            device_str = device if device is not None else 'None'

            generated_code = f"""
            from pyvqnet.tensor import linspace

            # 生成等间距序列
            start = {start if start is not None else 0}
            end = {end if end is not None else 10}
            steps = {steps if steps is not None else 5}

            result = linspace(start, end, steps, dtype={dtype_str}, device={device_str})
            print(f"生成从{start}到{end}共{steps}个点的等间距序列:")
            print(result)
            """

            return {
                "status": "success",
                "message": f"已生成linspace等间距序列代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet linspace API生成的真实代码，生成等间距序列。"
            }
        elif "pyvqnet.tensor.logical_and" in tool_name or "logical_and" in tool_name:
            # logical_and 逻辑与
            t1 = arguments.get('t1')
            t2 = arguments.get('t2')

            generated_code = f"""
            from pyvqnet.tensor import logical_and, tensor

            # 创建输入张量
            t1 = {t1 if t1 else 'tensor([True, False, True, False])'}
            t2 = {t2 if t2 else 'tensor([True, True, False, False])'}

            # 逻辑与操作
            result = logical_and(t1, t2)
            print("输入1:", t1)
            print("输入2:", t2)
            print("逻辑与结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成logical_and逻辑与代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet logical_and API生成的真实代码，逐元素逻辑与操作。"
            }
        elif "pyvqnet.tensor.logical_not" in tool_name or "logical_not" in tool_name:
            # logical_not 逻辑非
            t = arguments.get('t')

            generated_code = f"""
            from pyvqnet.tensor import logical_not, tensor

            # 创建输入张量
            t = {t if t else 'tensor([True, False, 1, 0])'}

            # 逻辑非操作
            result = logical_not(t)
            print("输入:", t)
            print("逻辑非结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成logical_not逻辑非代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet logical_not API生成的真实代码，逐元素逻辑非操作。"
            }
        elif "pyvqnet.tensor.logical_or" in tool_name or "logical_or" in tool_name:
            # logical_or 逻辑或
            t1 = arguments.get('t1')
            t2 = arguments.get('t2')

            generated_code = f"""
            from pyvqnet.tensor import logical_or, tensor

            # 创建输入张量
            t1 = {t1 if t1 else 'tensor([True, False, True, False])'}
            t2 = {t2 if t2 else 'tensor([True, True, False, False])'}

            # 逻辑或操作
            result = logical_or(t1, t2)
            print("输入1:", t1)
            print("输入2:", t2)
            print("逻辑或结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成logical_or逻辑或代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet logical_or API生成的真实代码，逐元素逻辑或操作。"
            }
        elif "pyvqnet.tensor.logical_xor" in tool_name or "logical_xor" in tool_name:
            # logical_xor 逻辑异或
            t1 = arguments.get('t1')
            t2 = arguments.get('t2')

            generated_code = f"""
            from pyvqnet.tensor import logical_xor, tensor

            # 创建输入张量
            t1 = {t1 if t1 else 'tensor([True, False, True, False])'}
            t2 = {t2 if t2 else 'tensor([True, True, False, False])'}

            # 逻辑异或操作
            result = logical_xor(t1, t2)
            print("输入1:", t1)
            print("输入2:", t2)
            print("逻辑异或结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成logical_xor逻辑异或代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet logical_xor API生成的真实代码，逐元素逻辑异或操作。"
            }
        elif "pyvqnet.tensor.logspace" in tool_name or "logspace" in tool_name:
            # logspace 生成对数间距序列
            start = arguments.get('start')
            end = arguments.get('end')
            steps = arguments.get('steps')
            base = arguments.get('base', 10.0)
            dtype = arguments.get('dtype', None)
            device = arguments.get('device', None)

            dtype_str = dtype if dtype is not None else 'None'
            device_str = device if device is not None else 'None'

            generated_code = f"""
            from pyvqnet.tensor import logspace

            # 生成对数间距序列
            start = {start if start is not None else 0}
            end = {end if end is not None else 3}
            steps = {steps if steps is not None else 4}
            base = {base}

            result = logspace(start, end, steps, base=base, dtype={dtype_str}, device={device_str})
            print(f"生成从10^{start}到10^{end}共{steps}个点的对数间距序列:")
            print(result)
            """

            return {
                "status": "success",
                "message": f"已生成logspace对数间距序列代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet logspace API生成的真实代码，生成对数间距序列。"
            }
        elif "pyvqnet.tensor.masked_fill" in tool_name or "masked_fill" in tool_name:
            # masked_fill 掩码填充
            t = arguments.get('t')
            mask = arguments.get('mask')
            value = arguments.get('value')

            generated_code = f"""
            from pyvqnet.tensor import masked_fill, tensor

            # 创建输入张量和掩码
            t = {t if t else 'tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])'}
            mask = {mask if mask else 'tensor([[True, False, True], [False, True, False], [True, False, True]])'}
            value = {value if value is not None else 0}

            # 掩码填充
            result = masked_fill(t, mask, value)
            print("输入张量:", t)
            print("掩码:", mask)
            print(f"填充值: {value}")
            print("结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成masked_fill掩码填充代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet masked_fill API生成的真实代码，根据掩码填充张量。"
            }
        elif "pyvqnet.tensor.minimum" in tool_name or "minimum" in tool_name:
            # minimum 逐元素最小值
            t1 = arguments.get('t1')
            t2 = arguments.get('t2')

            generated_code = f"""
            from pyvqnet.tensor import minimum, tensor

            # 创建输入张量
            t1 = {t1 if t1 else 'tensor([1, 3, 5, 7])'}
            t2 = {t2 if t2 else 'tensor([2, 2, 6, 6])'}

            # 逐元素最小值
            result = minimum(t1, t2)
            print("输入1:", t1)
            print("输入2:", t2)
            print("逐元素最小值:", result)
            """

            return {
                "status": "success",
                "message": f"已生成minimum逐元素最小值代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet minimum API生成的真实代码，逐元素取最小值。"
            }
        elif "pyvqnet.tensor.moveaxis" in tool_name or "moveaxis" in tool_name:
            # moveaxis 移动轴
            t = arguments.get('t')
            source = arguments.get('source')
            destination = arguments.get('destination')

            source_str = str(source) if source is not None else '0'
            dest_str = str(destination) if destination is not None else '-1'

            generated_code = f"""
            from pyvqnet.tensor import moveaxis, tensor

            # 创建输入张量
            t = {t if t else 'tensor.randn([2, 3, 4, 5])'}
            source = {source_str}
            destination = {dest_str}

            # 移动轴
            result = moveaxis(t, source, destination)
            print("输入形状:", t.shape)
            print(f"移动轴 {source} 到 {destination}")
            print("输出形状:", result.shape)
            """

            return {
                "status": "success",
                "message": f"已生成moveaxis移动轴代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet moveaxis API生成的真实代码，移动张量轴到指定位置。"
            }
        elif "pyvqnet.tensor.nonzero" in tool_name or "nonzero" in tool_name:
            # nonzero 获取非零元素索引
            t = arguments.get('t')

            generated_code = f"""
            from pyvqnet.tensor import nonzero, tensor

            # 创建输入张量
            t = {t if t else 'tensor([[0, 1, 0], [2, 0, 3], [0, 4, 0]])'}

            # 获取非零索引
            indices = nonzero(t)
            print("输入张量:", t)
            print("非零元素索引:", indices)
            """

            return {
                "status": "success",
                "message": f"已生成nonzero非零元素索引代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet nonzero API生成的真实代码，获取非零元素的索引。"
            }
        elif "pyvqnet.tensor.not_equal" in tool_name or "not_equal" in tool_name:
            # not_equal 不相等比较
            t1 = arguments.get('t1')
            t2 = arguments.get('t2')

            generated_code = f"""
            from pyvqnet.tensor import not_equal, tensor

            # 创建输入张量
            t1 = {t1 if t1 else 'tensor([1, 2, 3, 4])'}
            t2 = {t2 if t2 else 'tensor([1, 3, 3, 5])'}

            # 不相等比较
            result = not_equal(t1, t2)
            print("输入1:", t1)
            print("输入2:", t2)
            print("不相等比较结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成not_equal不相等比较代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet not_equal API生成的真实代码，逐元素比较不相等。"
            }
        elif "pyvqnet.tensor.ones_like" in tool_name or "ones_like" in tool_name:
            # ones_like 生成同形状全1张量
            t = arguments.get('t')
            dtype = arguments.get('dtype', None)
            device = arguments.get('device', None)

            dtype_str = dtype if dtype is not None else 'None'
            device_str = device if device is not None else 'None'

            generated_code = f"""
            from pyvqnet.tensor import ones_like, tensor

            # 创建输入张量
            t = {t if t else 'tensor([[1, 2, 3], [4, 5, 6]])'}

            # 生成同形状全1张量
            result = ones_like(t, dtype={dtype_str}, device={device_str})
            print("输入张量:", t)
            print("输入形状:", t.shape)
            print("结果:", result)
            print("结果形状:", result.shape)
            """

            return {
                "status": "success",
                "message": f"已生成ones_like同形状全1张量代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet ones_like API生成的真实代码，生成与输入同形状的全1张量。"
            }
        elif "pyvqnet.tensor.pack_pad_sequence" in tool_name or "pack_pad_sequence" in tool_name:
            # pack_pad_sequence 打包填充序列
            sequences = arguments.get('sequences')
            batch_first = arguments.get('batch_first', False)
            padding_value = arguments.get('padding_value', 0.0)

            sequences_str = str(sequences) if sequences else '[tensor([1, 2]), tensor([3, 4, 5]), tensor([6])]'
            generated_code = f"""
            from pyvqnet.tensor import pack_pad_sequence, tensor

            # 创建序列列表
            sequences = {sequences_str}

            # 打包填充序列
            packed_seq, lengths = pack_pad_sequence(
                sequences,
                batch_first={batch_first},
                padding_value={padding_value}
            )
            print("打包后的序列:", packed_seq)
            print("序列长度:", lengths)
            """

            return {
                "status": "success",
                "message": f"已生成pack_pad_sequence打包填充序列代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet pack_pad_sequence API生成的真实代码，打包填充可变长度序列。"
            }
        elif "pyvqnet.tensor.pad_packed_sequence" in tool_name or "pad_packed_sequence" in tool_name:
            # pad_packed_sequence 解包填充序列
            packed_seq = arguments.get('packed_seq')
            lengths = arguments.get('lengths')
            batch_first = arguments.get('batch_first', False)
            padding_value = arguments.get('padding_value', 0.0)
            total_length = arguments.get('total_length', None)

            total_len_str = str(total_length) if total_length is not None else 'None'
            generated_code = f"""
            from pyvqnet.tensor import pack_pad_sequence, pad_packed_sequence, tensor

            # 先创建打包序列
            sequences = [tensor([1, 2]), tensor([3, 4, 5]), tensor([6])]
            packed_seq, lengths = pack_pad_sequence(sequences)

            # 解包填充序列
            padded_seq, _ = pad_packed_sequence(
                packed_seq,
                lengths,
                batch_first={batch_first},
                padding_value={padding_value},
                total_length={total_len_str}
            )
            print("解包后的填充序列:", padded_seq)
            print("形状:", padded_seq.shape)
            """

            return {
                "status": "success",
                "message": f"已生成pad_packed_sequence解包填充序列代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet pad_packed_sequence API生成的真实代码，解包填充打包的序列。"
            }
        elif "pyvqnet.tensor.pad_sequence" in tool_name or "pad_sequence" in tool_name:
            # pad_sequence 填充序列
            sequences = arguments.get('sequences')
            batch_first = arguments.get('batch_first', False)
            padding_value = arguments.get('padding_value', 0.0)

            sequences_str = str(sequences) if sequences else '[tensor([1, 2]), tensor([3, 4, 5]), tensor([6])]'
            generated_code = f"""
            from pyvqnet.tensor import pad_sequence, tensor

            # 创建序列列表
            sequences = {sequences_str}

            # 填充序列
            padded_seq = pad_sequence(
                sequences,
                batch_first={batch_first},
                padding_value={padding_value}
            )
            print("填充后的序列:", padded_seq)
            print("形状:", padded_seq.shape)
            """

            return {
                "status": "success",
                "message": f"已生成pad_sequence填充序列代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet pad_sequence API生成的真实代码，填充可变长度序列到相同长度。"
            }
        elif "pyvqnet.tensor.scatter" in tool_name or "scatter" in tool_name:
            # scatter 散布
            t = arguments.get('t')
            dim = arguments.get('dim', 0)
            index = arguments.get('index')
            src = arguments.get('src')

            generated_code = f"""
            from pyvqnet.tensor import scatter, tensor, zeros

            # 创建输入张量
            t = {t if t else 'zeros([3, 5])'}
            dim = {dim}
            index = {index if index else 'tensor([[0, 1, 2], [0, 1, 2]])'}
            src = {src if src else 'tensor([[1, 2, 3], [4, 5, 6]])'}

            # 散布操作
            result = scatter(t, dim, index, src)
            print("输入张量:", t)
            print("索引:", index)
            print("源张量:", src)
            print(f"维度: {dim}")
            print("散布结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成scatter散布代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet scatter API生成的真实代码，沿指定维度散布源张量值到输入张量。"
            }
        elif "pyvqnet.tensor.select" in tool_name or "select" in tool_name:
            # select 选择
            t = arguments.get('t')
            dim = arguments.get('dim', 0)
            index = arguments.get('index')

            generated_code = f"""
            from pyvqnet.tensor import select, tensor

            # 创建输入张量
            t = {t if t else 'tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])'}
            dim = {dim}
            index = {index if index is not None else 1}

            # 选择操作
            result = select(t, dim, index)
            print("输入张量:", t)
            print(f"选择维度 {dim} 的索引 {index}")
            print("选择结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成select选择代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet select API生成的真实代码，沿指定维度选择元素。"
            }
        elif "pyvqnet.tensor.stack" in tool_name or "stack" in tool_name:
            # stack 堆叠
            tensors = arguments.get('tensors')
            dim = arguments.get('dim', 0)

            tensors_str = str(tensors) if tensors else '[tensor([1, 2]), tensor([3, 4]), tensor([5, 6])]'
            generated_code = f"""
            from pyvqnet.tensor import stack, tensor

            # 创建张量列表
            tensors = {tensors_str}

            # 堆叠操作
            result = stack(tensors, dim={dim})
            print("输入张量:", tensors)
            print(f"堆叠维度: {dim}")
            print("堆叠结果:", result)
            print("结果形状:", result.shape)
            """

            return {
                "status": "success",
                "message": f"已生成stack堆叠代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet stack API生成的真实代码，沿新维度堆叠张量列表。"
            }
        elif "pyvqnet.tensor.swapaxis" in tool_name or "swapaxis" in tool_name:
            # swapaxis 交换轴
            t = arguments.get('t')
            axis1 = arguments.get('axis1', 0)
            axis2 = arguments.get('axis2', 1)

            generated_code = f"""
            from pyvqnet.tensor import swapaxis, tensor

            # 创建输入张量
            t = {t if t else 'tensor.randn([2, 3, 4, 5])'}
            axis1 = {axis1}
            axis2 = {axis2}

            # 交换轴
            result = swapaxis(t, axis1, axis2)
            print("输入形状:", t.shape)
            print(f"交换轴 {axis1} 和 {axis2}")
            print("输出形状:", result.shape)
            """

            return {
                "status": "success",
                "message": f"已生成swapaxis交换轴代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet swapaxis API生成的真实代码，交换张量的两个轴。"
            }
        elif "pyvqnet.tensor.tile" in tool_name or "tile" in tool_name:
            # tile 平铺
            t = arguments.get('t')
            reps = arguments.get('reps')

            reps_str = str(reps) if reps else '(2, 3)'
            generated_code = f"""
            from pyvqnet.tensor import tile, tensor

            # 创建输入张量
            t = {t if t else 'tensor([[1, 2], [3, 4]])'}
            reps = {reps_str}

            # 平铺操作
            result = tile(t, reps)
            print("输入张量:", t)
            print("输入形状:", t.shape)
            print(f"平铺倍数: {reps}")
            print("结果:", result)
            print("结果形状:", result.shape)
            """

            return {
                "status": "success",
                "message": f"已生成tile平铺代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet tile API生成的真实代码，按指定倍数平铺张量。"
            }
        elif "pyvqnet.tensor.to_tensor" in tool_name or "to_tensor" in tool_name:
            # to_tensor 转换为张量
            data = arguments.get('data')
            dtype = arguments.get('dtype', None)
            device = arguments.get('device', None)
            requires_grad = arguments.get('requires_grad', False)

            dtype_str = dtype if dtype is not None else 'None'
            device_str = device if device is not None else 'None'
            data_str = str(data) if data else '[1, 2, 3, 4, 5]'

            generated_code = f"""
            from pyvqnet.tensor import to_tensor

            # 转换为张量
            data = {data_str}
            t = to_tensor(
                data,
                dtype={dtype_str},
                device={device_str},
                requires_grad={requires_grad}
            )
            print("输入数据:", data)
            print("转换后的张量:", t)
            print("张量形状:", t.shape)
            """

            return {
                "status": "success",
                "message": f"已生成to_tensor转换代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet to_tensor API生成的真实代码，将Python列表或numpy数组转换为QTensor。"
            }
        elif "pyvqnet.tensor.unsqueeze" in tool_name or "unsqueeze" in tool_name:
            # unsqueeze 增加维度
            t = arguments.get('t')
            dim = arguments.get('dim', 0)

            generated_code = f"""
            from pyvqnet.tensor import unsqueeze, tensor

            # 创建输入张量
            t = {t if t else 'tensor([1, 2, 3])'}
            dim = {dim}

            # 增加维度
            result = unsqueeze(t, dim)
            print("输入张量:", t)
            print("输入形状:", t.shape)
            print(f"在维度 {dim} 增加新轴")
            print("结果:", result)
            print("结果形状:", result.shape)
            """

            return {
                "status": "success",
                "message": f"已生成unsqueeze增加维度代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet unsqueeze API生成的真实代码，在指定位置增加新维度。"
            }
        elif "pyvqnet.tensor.where" in tool_name or "where" in tool_name:
            # where 条件选择
            condition = arguments.get('condition')
            x = arguments.get('x')
            y = arguments.get('y')

            generated_code = f"""
            from pyvqnet.tensor import where, tensor

            # 创建输入
            condition = {condition if condition else 'tensor([True, False, True, False])'}
            x = {x if x else 'tensor([1, 2, 3, 4])'}
            y = {y if y else 'tensor([10, 20, 30, 40])'}

            # 条件选择
            result = where(condition, x, y)
            print("条件:", condition)
            print("x:", x)
            print("y:", y)
            print("条件选择结果:", result)
            """

            return {
                "status": "success",
                "message": f"已生成where条件选择代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet where API生成的真实代码，根据条件选择x或y中的元素。"
            }
        elif "pyvqnet.tensor.zeros_like" in tool_name or "zeros_like" in tool_name:
            # zeros_like 生成同形状全0张量
            t = arguments.get('t')
            dtype = arguments.get('dtype', None)
            device = arguments.get('device', None)

            dtype_str = dtype if dtype is not None else 'None'
            device_str = device if device is not None else 'None'

            generated_code = f"""
            from pyvqnet.tensor import zeros_like, tensor

            # 创建输入张量
            t = {t if t else 'tensor([[1, 2, 3], [4, 5, 6]])'}

            # 生成同形状全0张量
            result = zeros_like(t, dtype={dtype_str}, device={device_str})
            print("输入张量:", t)
            print("输入形状:", t.shape)
            print("结果:", result)
            print("结果形状:", result.shape)
            """

            return {
                "status": "success",
                "message": f"已生成zeros_like同形状全0张量代码",
                "generated_code": generated_code,
                "parameters": arguments,
                "note": "这是基于pyVQNet zeros_like API生成的真实代码，生成与输入同形状的全0张量。"
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