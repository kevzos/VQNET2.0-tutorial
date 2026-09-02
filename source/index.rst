.. VQNet documentation master file, created by
   sphinx-quickstart on Tue Jul 27 15:25:07 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

VQNet
=================================


 
VQNet 是本源量子自主研发的量子机器学习计算框架，以参数化量子线路（Parameterized Quantum Circuit, PQC）为核心计算原语，将其作为可微算子嵌入经典神经网络，依托自动微分机制实现混合量子-经典模型的端到端构建与梯度反传，并支持调用本源量子计算机与量子云服务完成线路模拟与真实芯片实验。VQNet 提供自研的高性能张量库 ``pyvqnet.tensor``（100+ 计算接口，支持自动微分）与经典神经网络模块 ``pyvqnet.nn``（卷积、池化、循环神经网络、Transformer 注意力、多种归一化与优化器等），可与量子计算接口组合搭建任意混合架构模型。

量子计算层面，VQNet 提供双模拟后端：基于状态向量演化的 ``QMachine`` 接口（``pyvqnet.qnn.vqc``，含 50+ 量子逻辑门、``MeasureAll``/``Probability``/``Samples`` 等测量族、``QuantumLayerAdjoint`` 伴随梯度接口，以及 UCCSD、QSVT、硬件高效拟设等线路模板），和面向较大比特规模线路的张量网络/矩阵乘积态后端（``pyvqnet.qnn.vqc.tn``，提供 ``native`` 与 ``torch`` 两套实现）。框架内置线路编译优化 pass（旋转门合并、受控门交换等）与 OriginIR 线路导入导出；同时提供 ``pyvqnet.qnn`` 下的 QLSTM、QRNN、QMLP、Quanvolution 等量子-经典融合层，以及基于 pyqpanda3 与量子云的 pq3 系列量子层（含异步批量云端提交），打通从模拟到真实硬件的迁移路径。

工程层面，VQNet 内置基于 gloo/NCCL 集合通信的多进程分布式运行时（``vqnetrun`` 启动器），支持数据并行、张量并行（``ColumnParallelLinear``/``RowParallelLinear``）、流水线并行及三者混合的并行策略（``ParallelTrainingWrapper``），并提供 ``DistributedQMachine``/``DistQuantumLayerAdjoint`` 支撑多进程量子线路梯度计算；针对大模型场景，提供 RoPE、SwiGLU、融合 MoE、缩放 Softmax、top-k/top-p/min-p 采样等算子与 ``pyvqnet.torch.trl`` 微调损失族（SFT/DPO/PPO/GRPO/Reward），配合 peft/LlamaFactory 实现基于量子线路的大模型微调。

本使用文档是VQNet的 API 以及示例文档。英文版本可见:  `VQNet API DOC. <https://vqnet20-tutorial-en.readthedocs.io/en/latest/>`_


VQNet 的核心特点
-----------------

多平台兼容性与跨环境支持
~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet 支持用户在多种硬件和操作系统环境中进行量子与经典机器学习的研究与开发。无论是使用 CPU、GPU 进行量子计算模拟，还是通过本源量子云服务调用真实量子芯片，VQNet 都能提供无缝支持。目前，VQNet 已兼容 Windows (x86)、Linux (x86) 和 macOS (ARM) 系统的 Python 3.10～Python 3.14 版本。

完善的接口设计与易用性
~~~~~~~~~~~~~~~~~~~~~~~

VQNet 采用 Python 作为前端语言，提供类似于主流神经网络训练框架的编程接口，并可支持选择包括 ``torch`` 等多种计算后端运行经典与量子机器学习模型。框架内置了：100+ 常用 Tensor 计算接口、100+ 量子变分线路计算接口、50+ 经典神经网络接口。这些接口覆盖了从经典机器学习到量子机器学习的完整开发流程，并且将持续更新。

高效的计算性能与扩展能力
~~~~~~~~~~~~~~~~~~~~~~~~~

- **真实量子芯片实验支持**：对于需要真实量子芯片实验的用户，VQNet 集成了 本源 pyqpanda3 接口，并结合本源司南的高效调度能力，可实现快速的量子线路模拟计算和真实芯片运行。
- **本地计算优化**：对于本地计算需求，VQNet 提供基于 CPU 或 GPU 的量子机器学习编程接口，利用自动微分技术进行量子变分线路梯度计算。相比传统参数移位法，梯度计算效率提升明显，性能测试可见 :ref:`benchmarks` 。
- **分布式计算支持**：VQNet 支持基于 gloo/nccl 的分布式计算，可在多个节点上实现训练分布式混合量子-经典神经网络模型的功能。

丰富的应用场景与示例支持
~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet 不仅是一个强大的开发工具，还在公司内部多个项目中得到了广泛应用，包括 电力优化、医疗数据分析、图像处理 等领域。为了帮助用户快速上手，VQNet 在官网和 API 在线文档中提供了涵盖从基础教程到高级应用的多种场景。这些资源让用户能够轻松理解如何利用 VQNet 解决实际问题，并快速构建自己的量子机器学习应用。



.. toctree::
    :caption: 安装介绍
    :maxdepth: 2

    rst/install.rst

.. toctree::
    :caption: 上手实例
    :maxdepth: 2

    rst/vqc_demo.rst
    rst/qml_demo.rst

.. toctree::
    :caption: 经典神经网络接口介绍
    :maxdepth: 2

    rst/QTensor.rst
    rst/nn.rst
    rst/utils.rst

.. toctree::
    :caption: 使用pyqpanda3的量子神经网络接口
    :maxdepth: 2


    rst/qnn_pq3.rst

.. toctree::
    :caption: 量子神经网络自动微分模拟接口
    :maxdepth: 2

    rst/vqc.rst


.. toctree::
    :caption: 量子大模型微调
    :maxdepth: 2

    rst/llm.rst

.. toctree::
    :caption: 其他
    :maxdepth: 2
    
    rst/vqnet_dist.rst
    rst/torch_api.rst
    rst/FAQ.rst
    rst/CHANGELOG.rst





