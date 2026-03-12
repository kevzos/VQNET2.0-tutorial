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