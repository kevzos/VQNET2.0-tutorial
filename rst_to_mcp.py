"""
从 RST 文档自动解析并生成 MCP 工具定义
适用于 Sphinx 文档系统的 RST 格式
"""

import re
import json
from typing import Dict, List, Optional, Any


class RSTParser:
    """RST文档解析器"""

    def __init__(self):
        self.current_line = 0
        self.lines = []
        self.current_class_full_name = None  # Track current class context for methods

    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """解析RST文件，提取所有类/函数定义"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        self.lines = content.split('\n')
        self.current_line = 0

        definitions = []

        while self.current_line < len(self.lines):
            line = self.lines[self.current_line]
            stripped = line.strip()

            # 查找类定义
            if stripped.startswith('.. py:class::'):
                class_def = self._parse_class_definition()
                if class_def:
                    definitions.append(class_def)

            # 查找函数定义
            elif stripped.startswith('.. py:function::'):
                func_def = self._parse_function_definition()
                if func_def:
                    definitions.append(func_def)

            # 查找方法定义
            elif stripped.startswith('.. py:method::'):
                method_def = self._parse_method_definition()
                if method_def:
                    definitions.append(method_def)

            # 查找属性定义
            elif stripped.startswith('.. py:attribute::'):
                # Treat attribute as a method with no parameters
                attr_def = self._parse_class_definition(def_type='attribute')
                if attr_def:
                    definitions.append(attr_def)

            self.current_line += 1

        return definitions

    def _parse_class_definition(self, def_type: str = "class") -> Optional[Dict[str, Any]]:
        """解析类定义"""
        start_line = self.current_line
        line = self.lines[self.current_line].rstrip()

        # 提取类/函数/方法/属性签名
        # 格式: .. py:class:: module.path.ClassName(param1, param2=default, ...)
        # 格式: .. py:function:: module.path.func_name(param1, param2=default, ...)
        # 格式: .. py:method:: module.path.method_name(param1, param2=default, ...)
        # 格式: .. py:attribute:: attribute_name
        # 可能有多行，先获取完整签名
        signature = line.replace('.. py:class::', '', 1).replace('.. py:function::', '', 1).replace('.. py:method::', '', 1).replace('.. py:attribute::', '', 1).strip()

        # 如果签名不完整（没有括号），检查下一行
        if '(' not in signature and self.current_line + 1 < len(self.lines):
            next_line = self.lines[self.current_line + 1].rstrip()
            if next_line and not next_line.startswith(' ') and not next_line.startswith('\t'):
                # 下一行不是缩进，可能是参数部分
                pass
            else:
                # 可能签名被分割到多行
                # 暂时只处理单行
                pass

        # 解析模块路径和类名
        # 格式: module.path.ClassName(params)
        if '(' in signature:
            before_paren = signature[:signature.index('(')]
            params_str = signature[signature.index('(')+1:signature.rindex(')')]
        else:
            before_paren = signature
            params_str = ""

        # 提取类名（最后一个点之后的部分）
        if '.' in before_paren:
            module_path = before_paren[:before_paren.rindex('.')]
            class_name = before_paren[before_paren.rindex('.')+1:]
        else:
            module_path = ""
            class_name = before_paren

        # 解析参数
        params = self._parse_parameter_string(params_str)

        # 移动到签名后的行
        self.current_line += 1

        # 收集描述和参数文档
        description_lines = []
        param_docs = {}
        return_doc = ""

        # 首先收集描述（在 :param: 之前的所有非空行）
        while self.current_line < len(self.lines):
            line = self.lines[self.current_line].rstrip()

            # 检查是否开始参数节或新定义
            if (':param ' in line or ':type ' in line or ':return:' in line or
                ':raises ' in line or line.strip().startswith('.. note::') or
                line.strip().startswith('.. warning::') or
                line.strip().startswith('.. ')):
                break

            # 检查是否开始新节或定义
            if (line.strip().startswith('.. py:') or
                line.strip().startswith('.. _') or
                (line.strip() and not line.startswith(' ') and not line.startswith('\t'))):
                # 检查是否是新节标题
                if self.current_line + 1 < len(self.lines):
                    next_line = self.lines[self.current_line + 1].rstrip()
                    if re.match(r'^[=*-]{3,}$', next_line):
                        break

            if line.strip() and not line.startswith('    ') and not line.startswith('\t'):
                # 描述文本
                description_lines.append(line.strip())

            self.current_line += 1

        # 现在收集参数文档
        while self.current_line < len(self.lines):
            line = self.lines[self.current_line].rstrip()

            # 检查是否开始新的定义或节
            if (line.strip().startswith('.. py:') or
                line.strip().startswith('.. _') or
                (line.strip() and not line.startswith(' ') and not line.startswith('\t'))):
                # 检查是否是新节标题
                if self.current_line + 1 < len(self.lines):
                    next_line = self.lines[self.current_line + 1].rstrip()
                    # 简单的节标题检测：如果下一行是等号、星号或减号组成的行
                    if next_line and all(c in '=-*' for c in next_line.strip()) and len(next_line.strip()) >= 3:
                        break
                # 如果是新的定义行，直接停止
                if line.strip().startswith('.. py:class::') or line.strip().startswith('.. py:function::') or line.strip().startswith('.. py:method::'):
                    break

            # 处理参数文档
            if ':param ' in line:
                # 提取参数名和描述
                param_match = re.match(r'.*:param\s+(\w+):\s*(.*)', line)
                if param_match:
                    param_name = param_match.group(1)
                    param_desc = param_match.group(2)
                    param_docs[param_name] = param_desc
                else:
                    # 尝试另一种格式
                    param_match = re.match(r':param\s+(\w+):\s*(.*)', line)
                    if param_match:
                        param_name = param_match.group(1)
                        param_desc = param_match.group(2)
                        param_docs[param_name] = param_desc

            elif ':type ' in line:
                # 类型信息可以合并到参数文档
                pass

            elif ':return:' in line:
                return_match = re.match(r'.*:return:\s*(.*)', line)
                if return_match:
                    return_doc = return_match.group(1)

            elif ':raises ' in line:
                # 异常信息暂时忽略
                pass

            elif line.strip().startswith('.. note::'):
                # 注释块，可以跳过或包含在描述中
                pass

            elif line.strip() and param_docs:
                # 可能是参数描述的续行
                # 追加到最后一个参数
                last_param = list(param_docs.keys())[-1]
                param_docs[last_param] += ' ' + line.strip()
            elif line.strip() and return_doc:
                # 返回描述的续行
                return_doc += ' ' + line.strip()

            self.current_line += 1

        # 因为我们已经移动到下一个定义，需要回退一行
        self.current_line -= 1

        description = ' '.join(description_lines)

        # If we're parsing a method and inside a class, prepend class full name
        if def_type == 'method' and self.current_class_full_name and (not module_path or module_path == ""):
            # Method inside an existing class - combine full names
            full_name = f'{self.current_class_full_name}.{class_name}'
        elif module_path:
            full_name = f'{module_path}.{class_name}'
        else:
            full_name = class_name

        result = {
            'type': def_type,
            'module': module_path,
            'name': class_name,
            'full_name': full_name,
            'params': params,
            'description': description.strip(),
            'param_docs': param_docs,
            'return_doc': return_doc.strip()
        }

        # If this is a new class, update current_class_full_name
        if def_type == 'class' and full_name:
            self.current_class_full_name = full_name

        return result

    def _parse_function_definition(self) -> Optional[Dict[str, Any]]:
        """解析函数定义"""
        # 函数定义与类定义类似，使用相同的解析逻辑
        return self._parse_class_definition(def_type='function')

    def _parse_method_definition(self) -> Optional[Dict[str, Any]]:
        """解析方法定义"""
        # 方法定义与类定义类似，使用相同的解析逻辑
        return self._parse_class_definition(def_type='method')

    def _parse_parameter_string(self, params_str: str) -> List[str]:
        """解析参数字符串，返回参数列表"""
        if not params_str.strip():
            return []

        params = []
        current_param = ''
        paren_depth = 0
        bracket_depth = 0

        for char in params_str:
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == '[':
                bracket_depth += 1
            elif char == ']':
                bracket_depth -= 1
            elif char == ',' and paren_depth == 0 and bracket_depth == 0:
                params.append(current_param.strip())
                current_param = ''
                continue

            current_param += char

        if current_param.strip():
            params.append(current_param.strip())

        # 清理参数
        cleaned_params = []
        for param in params:
            # 移除多余的空白
            param = param.strip()
            if param:
                cleaned_params.append(param)

        return cleaned_params

    def generate_mcp_tool(self, definition: Dict[str, Any]) -> Dict[str, Any]:
        """从解析的数据生成 MCP 工具定义"""
        properties = {}
        required = []

        for param in definition['params']:
            # 提取参数名（如果有默认值）
            if '=' in param:
                param_parts = param.split('=')
                param_name = param_parts[0].strip()
                default_value = '='.join(param_parts[1:]).strip()
                has_default = True
            else:
                param_name = param.strip()
                default_value = None
                has_default = False

            # 清理参数名中的类型注解（如 diff_method:str -> diff_method）
            if ':' in param_name:
                param_name = param_name.split(':')[0].strip()

            # 简单的类型推断
            param_type = "string"  # 默认类型
            param_desc = definition['param_docs'].get(param_name, "")

            # 特定参数名检测
            if param_name == 'para_num' or param_name == 'shots':
                param_type = "integer"
            elif param_name == 'qvm_type':
                param_type = "string"
            elif param_name == 'pauli_str_dict':
                param_type = "object"  # dict或list，使用object
            elif param_name == 'dtype':
                param_type = "string"
            elif param_name == 'name':
                param_type = "string"
            elif param_name == 'initializer':
                param_type = "string"  # 可调用对象，用字符串表示
            elif param_name == 'origin_qprog_func':
                param_type = "string"  # 可调用对象，用字符串表示
            else:
                # 尝试从参数文档中推断类型
                if 'int' in param_desc.lower() or '整数' in param_desc:
                    param_type = "integer"
                elif 'float' in param_desc.lower() or 'number' in param_desc.lower() or '浮点数' in param_desc:
                    param_type = "number"
                elif 'bool' in param_desc.lower() or 'boolean' in param_desc.lower():
                    param_type = "boolean"
                elif 'list' in param_desc.lower() or 'array' in param_desc.lower() or '列表' in param_desc:
                    param_type = "array"
                elif 'dict' in param_desc.lower() or '字典' in param_desc:
                    param_type = "object"
                elif 'callable' in param_desc.lower() or '可调用' in param_desc:
                    param_type = "string"  # MCP不支持callable类型，用字符串表示

            # 检查参数名是否为空
            if not param_name:
                continue

            # 构建参数模式
            param_schema = {
                "type": param_type,
                "description": param_desc
            }

            # 添加枚举值
            if param_name == 'qvm_type' and 'cpu' in param_desc.lower() and 'gpu' in param_desc.lower():
                param_schema["enum"] = ["cpu", "gpu"]

            # 添加默认值
            if default_value is not None:
                # 处理特殊默认值
                if default_value == 'None':
                    param_schema["default"] = None
                elif default_value == '""' or default_value == "''":
                    param_schema["default"] = ""
                else:
                    # 尝试转换默认值为合适的类型
                    try:
                        if param_type == "integer":
                            param_schema["default"] = int(default_value)
                        elif param_type == "number":
                            param_schema["default"] = float(default_value)
                        elif param_type == "boolean":
                            if default_value.lower() in ['true', 'false']:
                                param_schema["default"] = default_value.lower() == 'true'
                            else:
                                param_schema["default"] = default_value
                        else:
                            param_schema["default"] = default_value.strip('"\'')
                    except (ValueError, TypeError):
                        param_schema["default"] = default_value.strip('"\'')

            properties[param_name] = param_schema

            # 如果没有默认值，则是必需的
            if not has_default and default_value is None:
                required.append(param_name)

        return {
            "name": definition['full_name'],
            "description": definition['description'],
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required if required else None
            }
        }


def main():
    """主函数"""
    parser = RSTParser()

    # 要处理的RST文件列表
    rst_files = [
        "source/rst/QTensor.rst",
        "source/rst/qnn_pq3.rst",
        "source/rst/vqc.rst",
        "source/rst/nn.rst",
        "source/rst/torch_api.rst"
    ]

    # 要提取的目标API列表
    target_apis = [
        # QTensor.rst
        "QTensor",
        "is_dense",
        "is_contiguous",
        "zero_grad",
        "backward",
        "to_numpy",
        "item",
        "contiguous",
        "argmax",
        "argmin",
        "fill_",
        "all",
        "any",
        "fill_rand_binary_",
        "fill_rand_signed_uniform_",
        "fill_rand_uniform_",
        "fill_rand_normal_",
        "ones",
        "ones_like",
        "zeros",
        "zeros_like",
        "full",
        "full_like",
        "arange",
        "linspace",
        "logspace",
        "eye",
        "diagonal",
        "diag",
        "randu",
        "randn",
        "binomial",
        "multinomial",
        "triu",
        "tril",
        "floor",
        "ceil",
        "round",
        "sort",
        "argsort",
        "topK",
        "argtopK",
        "add",
        "sub",
        "mul",
        "divide",
        "sums",
        "cumsum",
        "mean",
        "median",
        "std",
        "var",
        "matmul",
        "kron",
        "einsum",
        "reciprocal",
        "sign",
        "neg",
        "trace",
        "exp",
        "log",
        "log_softmax",
        "sqrt",
        "square",
        "abs",
        "power",
        "eigh",
        "frobenius_norm",
        "maximum",
        "minimum",
        "min",
        "max",
        "clip",
        "where",
        "nonzero",
        "isfinite",
        "isinf",
        "isnan",
        "isneginf",
        "isposinf",
        "logical_and",
        "logical_or",
        "logical_not",
        "logical_xor",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
        "equal",
        "not_equal",
        "bitwise_and",
        "broadcast",
        "select",
        "concatenate",
        "stack",
        "permute",
        "transpose",
        "tile",
        "swapaxis",
        "moveaxis",
        "masked_fill",
        "flatten",
        "reshape",
        "unsqueeze",
        "flip",
        "gather",
        "scatter",
        "broadcast_to",
        "to_tensor",
        "pad_sequence",
        "pad_packed_sequence",
        "pack_pad_sequence",
        # qnn_pq3.rst
        "QuantumLayer",
        "AmplitudeEmbeddingCircuit",
        "AngleEmbeddingCircuit",
        "grad",
        # vqc.rst
        "QMachine",
        "reset_states",
        # 基础量子门 - 函数式接口
        "hadamard",
        "t",
        "s",
        "paulix",
        "pauliy",
        "pauliz",
        "x1",
        # 基础量子门 - 类接口
        "I",
        "Hadamard",
        "T",
        "S",
        "PauliX",
        "PauliY",
        "PauliZ",
        "X1",
        # 测量操作
        "Measure",
        "Probability",
        "Samples",
        "HermitianExpval",
        "MeasureAll",
        # 双量子比特门
        "cnot",
        "swap",
        "toffoli",
        "cz",
        # 单量子比特参数门
        "rx",
        "ry",
        "rz",
        "RX",
        "RY",
        "RZ",
        # VQC模板
        "VQC_HardwareEfficientAnsatz",
        "VQC_AngleEmbedding",
        "VQC_RotCircuit",
        "VQC_VarMeasure",
        # 控制量子门
        "crx",
        "cry",
        "crz",
        "CRX",
        "CRY",
        "CRZ",
        # 高级VQC功能
        "VQC_AllSinglesDoubles",
        "VQC_BasisRotation",
        "VQC_QuantumPoolingCircuit",
        "vqc_qft_add_to_register",
        "VQC_FABLE",
        "VQC_LCU",
        "VQC_QSVT",
        "Quanvolution",
        "QDRL",
        "QGRU",
        "QLSTM",
        "QRLModel",
        "QRNN",
        "wrapper_single_qubit_op_fuse",
        # qnn_pq3.rst - 扩展
        "BasicEmbeddingCircuit",
        "IQPEmbeddingCircuits",
        "RotCircuit",
        "CRotCircuit",
        "expval",
        "QuantumMeasure",
        # qnn_pq3.rst - 量子层扩展
        "QpandaQProgVQCLayer",
        "QuantumLayerAdjoint",
        # qnn_pq3.rst - 量子门和模板
        "CSWAPcircuit",
        "Controlled_Hadamard",
        "CCZ",
        "HardwareEfficientAnsatz",
        "BasicEntanglerTemplate",
        "StronglyEntanglingTemplate",
        "ComplexEntangelingTemplate",
        "Quantum_Embedding",
        "QuantumPoolingCircuit",
        # qnn_pq3.rst - 费米子激发
        "FermionicSingleExcitation",
        "FermionicDoubleExcitation",
        # qnn_pq3.rst - 测量扩展
        "ProbsMeasure",
        "DensityMatrixFromQstate",
        "VN_Entropy",
        "Mutal_Info",
        "Purity",
        # nn.rst - 神经网络层
        "Linear",
        "Conv1d",
        "Conv2d",
        "ConvT2d",
        "BatchNorm1d",
        "BatchNorm2d",
        "LayerNorm1d",
        "LayerNorm2d",
        "LayerNormNd",
        "GroupNorm",
        "Embedding",
        "Dropout",
        "DropPath",
        "AvgPool1d",
        "MaxPool1d",
        "AvgPool2D",
        "MaxPool2D",
        "PixelShuffle",
        "PixelUnshuffle",
        "GRU",
        "RNN",
        "LSTM",
        "DynamicGRU",
        "DynamicRNN",
        "DynamicLSTM",
        "Interpolate",
        # nn.rst - 激活函数
        "ReLu",
        "LeakyReLu",
        "Sigmoid",
        "Gelu",
        "ELU",
        "Softplus",
        "Softsign",
        "Tanh",
        "Softmax",
        "HardSigmoid",
        # nn.rst - 损失函数
        "CrossEntropyLoss",
        "NLL_Loss",
        "MSELoss",
        "BCELoss",
        "SoftmaxCrossEntropy",
        # optim.rst - 优化器
        "Adam",
        "AdamW",
        "Adamax",
        "Adadelta",
        "Adagrad",
        "RMSProp",
        "SGD",
        "Rotosolve",
        # torch_api.rst - 后端配置
        "set_backend",
        "get_backend",
        # torch_api.rst - 基类
        "TorchModule",
        # torch_api.rst - 神经网络层
        "Linear",
        "Conv2D",
        # torch_api.rst - 激活函数
        "ReLu",
        "Sigmoid",
        "Softmax",
        "Tanh",
        "Gelu",
        "Softplus",
        # torch_api.rst - 损失函数
        "CrossEntropyLoss",
        "MeanSquaredError",
        # torch_api.rst - 其他层
        "Dropout",
        "BatchNorm2d",
        # torch_api.rst - VQC
        "QMachine",
        "vqc_basis_embedding",
        "vqc_angle_embedding",
        "vqc_controlled_hadamard"
    ]

    all_tools = []

    for rst_file in rst_files:
        try:
            print(f"\n处理文件: {rst_file}")
            definitions = parser.parse_file(rst_file)
            print(f"解析到 {len(definitions)} 个定义")

            # 查找目标API
            for defn in definitions:
                def_name = defn['name']
                if def_name in target_apis:
                    print(f"找到目标API: {defn['full_name']}")
                    # 生成MCP工具
                    tool = parser.generate_mcp_tool(defn)
                    all_tools.append(tool)

        except FileNotFoundError:
            print(f"文件未找到: {rst_file}")
            print("当前目录:", __file__)
        except Exception as e:
            print(f"解析错误: {e}")
            import traceback
            traceback.print_exc()

    # 保存所有工具到文件
    if all_tools:
        output_file = 'vqnet_all_tools.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_tools, f, indent=2, ensure_ascii=False)
        print(f"\n共生成 {len(all_tools)} 个工具定义，已保存到 {output_file}")

        # 打印生成的工具列表
        print("\n生成的工具列表:")
        for i, tool in enumerate(all_tools):
            print(f"{i+1}. {tool['name']}")
    else:
        print("\n未找到任何目标API")


if __name__ == "__main__":
    main()