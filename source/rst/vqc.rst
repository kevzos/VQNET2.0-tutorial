
.. _vqc_api:

变分量子线路自动微分模拟
***********************************

以下接口使用VQNet的QTensor接口构建常用量子逻辑门、量子线路以及测量方法,可使用自动微分模拟代替量子线路parameter-shift方法计算量子变分线路中参数的梯度。

我们可以像其他 `Module` 一样,使用 ``pyvqnet.qnn.vqc`` 模块下的接口构成量子与经典混合神经网络模型。在 `Module` 中需要定义虚拟机 `QMachine`,并且需要对machine的量子态矢数据 `states` 根据输入的batchsize进行reset_states。请具体看下例:

.. warning::

    为了在每次迭代运行量子线路前向函数保证量子态为初态，你需要在forward函数一开头运行 ``QMachine.reset_states(batchsize)``。
    
    为了支持批量数据运行量子线路，你将输入input编码到量子逻辑门上时候，input形状需要为[batchsize,*],其中*为要编码的逻辑门的参数个数。
    
    请参考如下例子：

.. code-block::

    from pyvqnet.nn import Module,Linear,ModuleList
    from pyvqnet.qnn.vqc import VQC_HardwareEfficientAnsatz,RZZ,RZ
    from pyvqnet.qnn.vqc import Probability,QMachine
    from pyvqnet import tensor

    class QM(Module):
        def __init__(self, name=""):
            super().__init__(name)
            self.linearx = Linear(4,2)
            self.ansatz = VQC_HardwareEfficientAnsatz(4, ["rx", "RY", "rz"],
                                        entangle_gate="cnot",
                                        entangle_rules="linear",
                                        depth=2)
            #基于VQC的RZ 在0比特上
            self.encode1 = RZ(wires=0)
            #基于VQC的RZ 在1比特上
            self.encode2 = RZ(wires=1)
            #基于VQC的概率测量 在0,2比特上
            self.measure = Probability(wires=[0,2])
            #量子设备QMachine,使用4个比特。
            self.device = QMachine(4)
        def forward(self, x, *args, **kwargs):
            #必须要将states reset到与输入一样的batchsize。
            self.device.reset_states(x.shape[0])
            y = self.linearx(x)
            #将输入编码到RZ门上,注意输入必须是 [batchsize,1]的shape
            self.encode1(params = y[:, 0],q_machine = self.device,)
            #将输入编码到RZ门上,注意输入必须是 [batchsize,1]的shape
            self.encode2(params = y[:, 1],q_machine = self.device,)
            self.ansatz(q_machine =self.device)
            return self.measure(q_machine =self.device)

    bz =3
    inputx = tensor.arange(1.0,bz*4+1).reshape([bz,4])
    inputx.requires_grad= True
    #像其他Module一样定义
    qlayer = QM()
    #前传
    y = qlayer(inputx)
    #反传
    y.backward()
    print(y)

如果要使用一些带训练参数的变分量子线路逻辑门,而不止像上例一样将数据编码到线路上,可以参考下面例子:

.. code-block::

    from pyvqnet.nn import Module,Linear,ModuleList
    from pyvqnet.qnn.vqc import VQC_HardwareEfficientAnsatz,RZZ,RZ,rz,ry,cnot
    from pyvqnet.qnn.vqc import Probability,QMachine
    from pyvqnet import tensor

    class QM(Module):
        def __init__(self, name=""):
            super().__init__(name)
            self.linearx = Linear(4,2)
            self.ansatz = VQC_HardwareEfficientAnsatz(4, ["rx", "RY", "rz"],
                                        entangle_gate="cnot",
                                        entangle_rules="linear",
                                        depth=2)
            #基于VQC的RZ 在0比特上
            self.encode1 = RZ(wires=0)
            #基于VQC的RZ 在1比特上
            self.encode2 = RZ(wires=1)
            #设置RZ trainable= True
            self.vqc = RZ(trainable = True,wires=1)
            #基于VQC的概率测量 在0,2比特上
            self.measure = Probability(wires=[0,2])
            #量子设备QMachine,使用4个比特。
            self.device = QMachine(4)
        def forward(self, x, *args, **kwargs):
            #必须要将states reset到与输入一样的batchsize。
            self.device.reset_states(x.shape[0])
            y = self.linearx(x)
            #将输入编码到RZ门上,注意输入必须是 [batchsize,1]的shape
            self.encode1(params = y[:, 0],q_machine = self.device,)
            #将输入编码到RZ门上,注意输入必须是 [batchsize,1]的shape
            self.encode2(params = y[:, 1],q_machine = self.device,)
            #使用RZ门构成的含参变分线路,会加入训练。
            self.vqc(q_machine =self.device)
            self.ansatz(q_machine =self.device)
            return self.measure(q_machine =self.device)

    bz =3
    inputx = tensor.arange(1.0,bz*4+1).reshape([bz,4])
    inputx.requires_grad= True
    #像其他Module一样定义
    qlayer = QM()
    #前传
    y = qlayer(inputx)
    #反传
    y.backward()
    print(y)


以下示例展示了如何在GPU上进行变分量子计算（包含数据编码与含参变分线路）：

.. code-block::

    from pyvqnet.nn import Module,Linear,ModuleList
    from pyvqnet.qnn.vqc import VQC_HardwareEfficientAnsatz,RZZ,RZ,rz,ry,cnot
    from pyvqnet.qnn.vqc import Probability,QMachine
    from pyvqnet import tensor
    from pyvqnet import DEV_GPU
    class QM(Module):
        def __init__(self, name=""):
            super().__init__(name)
            self.linearx = Linear(4,2)
            self.ansatz = VQC_HardwareEfficientAnsatz(4, ["rx", "RY", "rz"],
                                        entangle_gate="cnot",
                                        entangle_rules="linear",
                                        depth=2)
            #基于VQC的RZ 在0比特上
            self.encode1 = RZ(wires=0)
            #基于VQC的RZ 在1比特上
            self.encode2 = RZ(wires=1)
            #设置RZ trainable= True
            self.vqc = RZ(trainable = True,wires=1)
            #基于VQC的概率测量 在0,2比特上
            self.measure = Probability(wires=[0,2])
            #量子设备QMachine,使用4个比特。
            self.device = QMachine(4)
        def forward(self, x, *args, **kwargs):
            #必须要将states reset到与输入一样的batchsize。
            self.device.reset_states(x.shape[0])
            y = self.linearx(x)
            #将输入编码到RZ门上,注意输入必须是 [batchsize,1]的shape
            self.encode1(params = y[:, 0],q_machine = self.device,)
            #将输入编码到RZ门上,注意输入必须是 [batchsize,1]的shape
            self.encode2(params = y[:, 1],q_machine = self.device,)
            #使用RZ门构成的含参变分线路,会加入训练。
            self.vqc(q_machine =self.device)
            self.ansatz(q_machine =self.device)
            return self.measure(q_machine =self.device)

    bz =3
    #create tensor on GPU
    inputx = tensor.arange(1.0,bz*4+1,device=DEV_GPU).reshape([bz,4])
    inputx.requires_grad= True
    #像其他Module一样定义
    qlayer = QM()
    #move module to GPU
    qlayer = qlayer.to(DEV_GPU)
    #前传
    y = qlayer(inputx)
    #反传
    y.backward()
    print(y)

模拟器
=========================================

QMachine
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.QMachine(num_wires, dtype=pyvqnet.kcomplex64, save_ir=False, state_storage=None)

    变分量子计算的模拟器类,包含states属性为量子线路的statevectors。``states`` 是形状为 ``(batchsize, 2, ..., 2)`` 的 ``QTensor``,``dtype`` 为计算数据类型,``state_storage`` 为 ``states`` 的存储数据类型(未设置时为 ``None``)。

    .. note::

        在每次运行一个完整的量子线路之前,必须使用 ``pyvqnet.qnn.vqc.QMachine.reset_states(batchsize)`` 将模拟器里面初态重新初始化,并且广播为
        (batchsize,*) 维度从而适应批量数据训练。

    :param num_wires: 量子比特数。
    :param dtype: 计算数据的数据类型。默认值是pyvqnet.kcomplex64,对应的参数精度为pyvqnet.kfloat32;也可以设置为 ``pyvqnet.kcomplex128``,对应的参数精度为 ``pyvqnet.kfloat64``。传入 ``pyvqnet.kcomplex32`` 等价于 ``dtype=pyvqnet.kcomplex64`` 并且 ``state_storage=pyvqnet.kcomplex32``。
    :param save_ir: 设置为True时,将操作保存到originIR,默认值:False。
    :param state_storage: 态矢 ``states`` 的存储数据类型,默认值:None,表示与 ``dtype`` 相同。设置为 ``pyvqnet.kcomplex32``(或 ``torch.complex32``)时,``states`` 以半精度存储,每个振幅占4个字节;此时 ``dtype`` 只能是 ``pyvqnet.kcomplex64``,门内部的计算仍在 ``pyvqnet.kcomplex64`` 精度下进行,门的参数以 ``pyvqnet.kfloat16`` 存储,门矩阵为 ``pyvqnet.kcomplex32``,``MeasureAll`` 和 ``Probability`` 的测量结果的数据类型为 ``pyvqnet.kfloat16``。传入其他值会报 ``ValueError``。

    :return: 输出QMachine。

    Example::

        from pyvqnet.qnn.vqc import QMachine
        qm = QMachine(4)

        print(qm.states)
        print(qm.dtype, qm.state_storage, qm.num_wires, qm.batch_size)

        # [[[[[1.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]],
        #
        #    [[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]]],
        #
        #
        #   [[[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]],
        #
        #    [[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]]]]]
        # <QTensor [1, 2, 2, 2, 2] DEV_CPU complex64>
        # complex64 None 4 1

    ``state_storage=pyvqnet.kcomplex32`` 时,``dtype`` 仍为 ``pyvqnet.kcomplex64``,``states`` 以 ``pyvqnet.kcomplex32`` 存储,测量结果的数据类型为 ``pyvqnet.kfloat16``:

    .. code-block::

        from pyvqnet.qnn.vqc import QMachine, MeasureAll, ry
        from pyvqnet import kcomplex32, tensor

        # fp16 storage: the compute dtype stays kcomplex64, the states are complex32
        qm = QMachine(3, dtype=kcomplex32)
        print(qm.dtype, qm.state_storage, qm.states.dtype)

        x = tensor.QTensor([[0.3], [1.2]], requires_grad=True)
        qm.reset_states(2)
        ry(q_machine=qm, wires=0, params=x)
        y = MeasureAll(obs={"Z0": 1.0})(q_machine=qm)
        print(y, y.dtype)
        y.backward()
        print(x.grad)

        # complex64 complex32 complex32
        #
        # [[0.9556],
        #  [0.3618]]
        # <QTensor [2, 1] DEV_CPU float16> float16
        #
        # [[-0.2954102],
        #  [-0.9321289]]
        # <QTensor [2, 1] DEV_CPU float32>

    .. py:method:: reset_states(batchsize)

        将模拟器里面初态重新初始化,并且广播为
        (batchsize,[2]**num_qubits) 维度从而使得模拟器可以进行批量数据的模型计算。上一次前向的线路记录(``op_history``、originIR 字符串和参数登记)同时被清空。

        :param batchsize: 批处理的数量。

    .. py:method:: init_states(init_state)

        用给定的量子态设置模拟器的 ``states``,可以替代 ``reset_states`` 之后的默认初态 :math:`|00\cdots 0\rangle`。

        :param init_state: 归一化的量子态,``QTensor``,形状为 ``(batchsize, 2, ..., 2)``,或者 ``(2, ..., 2)``(沿 batch 维广播到每个样本)。每个样本的范数必须为1,否则报 ``ValueError``;数据类型必须与 ``dtype`` 相同(包括 ``state_storage=pyvqnet.kcomplex32`` 的情况,此时传入 ``pyvqnet.kcomplex64`` 的态),设备必须与模拟器相同。

        Example::

            from pyvqnet.qnn.vqc import QMachine, Probability
            from pyvqnet import tensor

            qm = QMachine(2)
            qm.reset_states(3)
            # one normalized state, broadcast over the 3 batch entries
            qm.init_states(tensor.QTensor([[0.0, 1.0], [0.0, 0.0]], dtype=tensor.kcomplex64))
            print(Probability(wires=[0, 1])(q_machine=qm))

            # one normalized state per batch entry
            qm.init_states(tensor.QTensor([[[0.0, 1.0], [0.0, 0.0]],
                                           [[0.0, 0.0], [1.0, 0.0]],
                                           [[0.0, 0.0], [0.0, 1.0]]], dtype=tensor.kcomplex64))
            print(Probability(wires=[0, 1])(q_machine=qm))

            # [[0.,1.,0.,0.],
            #  [0.,1.,0.,0.],
            #  [0.,1.,0.,0.]]
            # <QTensor [3, 4] DEV_CPU float32>
            #
            # [[0.,1.,0.,0.],
            #  [0.,0.,1.,0.],
            #  [0.,0.,0.,1.]]
            # <QTensor [3, 4] DEV_CPU float32>

    .. py:method:: is_adjoint()

        返回模拟器当前是否处于 adjoint(伴随)梯度模式。

        adjoint 模式不在 ``QMachine`` 上设置:把含有该模拟器的 ``Module`` 用 ``pyvqnet.qnn.vqc.QuantumLayerAdjoint`` 包装后,包装层在每次前向计算期间把模拟器切换到 adjoint 模式,前向计算结束后恢复为自动微分模式。

        :return: 处于 adjoint 模式时为 ``True``,否则为 ``False``。

        Example::

            import pyvqnet
            from pyvqnet import tensor
            from pyvqnet.qnn.vqc import QMachine, QuantumLayerAdjoint, RX, RY, CNOT, MeasureAll
            from pyvqnet.utils import set_random_seed

            set_random_seed(42)


            class QModel(pyvqnet.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.qm = QMachine(2)
                    self.rx = RX(trainable=False, wires=0)
                    self.ry = RY(trainable=True, wires=1)
                    self.cnot = CNOT(wires=[0, 1])
                    self.measure = MeasureAll(obs={"Z0": 1.0, "Z1": 1.0})
                    self.adjoint_flags = []

                def forward(self, x, *args, **kwargs):
                    self.adjoint_flags.append(self.qm.is_adjoint())
                    self.qm.reset_states(x.shape[0])
                    self.rx(params=x, q_machine=self.qm)
                    self.ry(q_machine=self.qm)
                    self.cnot(q_machine=self.qm)
                    return self.measure(q_machine=self.qm)


            x = tensor.QTensor([[0.5]], requires_grad=True)
            model = QModel()
            model(x)
            print(model.adjoint_flags, model.qm.is_adjoint())

            x = tensor.QTensor([[0.5]], requires_grad=True)
            adjoint_model = QuantumLayerAdjoint(model)
            adjoint_model.train()
            y = adjoint_model(x)
            print(model.adjoint_flags, model.qm.is_adjoint())
            y.backward()
            print(y, x.grad)

            # [False] False
            # [False, True] False
            #
            # [[1.2884717]]
            # <QTensor [1, 1] DEV_CPU float32>
            # [[-0.7038953]]
            # <QTensor [1, 1] DEV_CPU float32>

    .. py:method:: set_noise_seed(seed)

        设置该模拟器噪声信道蒙特卡洛采样(每条线路分支的抽取)的随机种子。

        :param seed: 随机种子;传入 ``None`` 表示使用新的随机熵。

        Example::

            from pyvqnet.qnn.vqc import MeasureAll, QMachine, depolarizing, ry

            qm = QMachine(1)
            qm.set_noise_seed(7)
            qm.reset_states(1024)            # 1024 trajectories
            ry(q_machine=qm, wires=0, params=0.4)
            depolarizing(q_machine=qm, wire=0, p=0.05)
            y = MeasureAll(obs={"Z0": 1.0})(q_machine=qm)
            print(y.to_numpy().mean())        # noisy <Z0>

            # 0.8724892


量子逻辑门接口
============================

本章的逻辑门函数共用同一套参数检查:``wires`` 中的量子比特索引不能重复,并且需要位于 ``[0, q_machine.num_wires)`` 内;含参逻辑门的 ``params`` 可以是 ``QTensor`` 或者 python 数值,``QTensor`` 的行数只能是1(同一个角度广播到所有样本)或者等于 ``q_machine.reset_states(batchsize)`` 的批处理数量。不满足时与比特索引不合法时一样报 ``ValueError``。

i
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.i(q_machine, wires, params=None, use_dagger=False)

    对 ``q_machine`` 中的态矢(statevectors)作用量子逻辑门 I 。

    :param q_machine: 量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import i,QMachine
        qm  = QMachine(4)
        i(q_machine=qm, wires=1,)
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]

        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]


        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]

        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]


I
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.I(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个I逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个I逻辑门类实例。

    Example::
        
        from pyvqnet.qnn.vqc import I,QMachine
        device = QMachine(4)
        layer = I(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)

hadamard
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.hadamard(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 hadamard 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import hadamard,QMachine
        qm  = QMachine(4)
        hadamard(q_machine=qm, wires=1,)
        print(qm.states)
        # [[[[[0.7071068+0.j 0.       +0.j]
        #     [0.       +0.j 0.       +0.j]]
        # 
        #    [[0.7071068+0.j 0.       +0.j]
        #     [0.       +0.j 0.       +0.j]]]
        # 
        # 
        #   [[[0.       +0.j 0.       +0.j]
        #     [0.       +0.j 0.       +0.j]]
        # 
        #    [[0.       +0.j 0.       +0.j]
        #     [0.       +0.j 0.       +0.j]]]]]


Hadamard
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.Hadamard(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个Hadamard逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 Hadamard逻辑门类实例。

    Example::
        
        from pyvqnet.qnn.vqc import Hadamard,QMachine
        device = QMachine(4)
        layer = Hadamard(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)



t
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.t(q_machine, wires, params=None, use_dagger=False)

    对q_machine中的态矢作用量子逻辑门 t 。T 门对 :math:`|1\rangle` 施加相位 :math:`e^{i\pi/4}`;``use_dagger=True`` 时作用其转置共轭 :math:`T^\dagger`。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。

    Example::

        from pyvqnet.qnn.vqc import QMachine, hadamard, t

        qm = QMachine(2)
        hadamard(q_machine=qm, wires=0)
        t(q_machine=qm, wires=0)
        print(qm.states)

        qm = QMachine(2)
        hadamard(q_machine=qm, wires=0)
        t(q_machine=qm, wires=0, use_dagger=True)
        print(qm.states)

        # [[[0.7071068+0.j ,0.       +0.j ],
        #   [0.5      +0.5j,0.       +0.j ]]]
        # <QTensor [1, 2, 2] DEV_CPU complex64>
        #
        # [[[0.7071068+0.j ,0.       +0.j ],
        #   [0.5      -0.5j,0.       +0.j ]]]
        # <QTensor [1, 2, 2] DEV_CPU complex64>

T
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.T(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个T逻辑门类 。``use_dagger=True`` 时该逻辑门为其转置共轭 :math:`T^\dagger`。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 T逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import QMachine, Hadamard, T

        device = QMachine(2)
        hadamard = Hadamard(wires=0)
        t_gate = T(wires=0)
        t_dagger = T(wires=0, use_dagger=True)

        device.reset_states(1)
        hadamard(q_machine=device)
        t_gate(q_machine=device)
        print(device.states)

        device.reset_states(1)
        hadamard(q_machine=device)
        t_dagger(q_machine=device)
        print(device.states)

        # [[[0.7071068+0.j ,0.       +0.j ],
        #   [0.5      +0.5j,0.       +0.j ]]]
        # <QTensor [1, 2, 2] DEV_CPU complex64>
        #
        # [[[0.7071068+0.j ,0.       +0.j ],
        #   [0.5      -0.5j,0.       +0.j ]]]
        # <QTensor [1, 2, 2] DEV_CPU complex64>

s
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.s(q_machine, wires, params=None, use_dagger=False)

    对q_machine中的态矢作用量子逻辑门 s 。S 门对 :math:`|1\rangle` 施加相位 :math:`i`;``use_dagger=True`` 时作用其转置共轭 :math:`S^\dagger`。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。

    Example::

        from pyvqnet.qnn.vqc import QMachine, hadamard, s

        qm = QMachine(2)
        hadamard(q_machine=qm, wires=0)
        s(q_machine=qm, wires=0)
        print(qm.states)

        qm = QMachine(2)
        hadamard(q_machine=qm, wires=0)
        s(q_machine=qm, wires=0, use_dagger=True)
        print(qm.states)

        # [[[0.7071068+0.j       ,0.       +0.j       ],
        #   [0.       +0.7071068j,0.       +0.j       ]]]
        # <QTensor [1, 2, 2] DEV_CPU complex64>
        #
        # [[[0.7071068+0.j       ,0.       +0.j       ],
        #   [0.       -0.7071068j,0.       +0.j       ]]]
        # <QTensor [1, 2, 2] DEV_CPU complex64>

S
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.S(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个S逻辑门类 。``use_dagger=True`` 时该逻辑门为其转置共轭 :math:`S^\dagger`。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 S逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import QMachine, Hadamard, S

        device = QMachine(2)
        hadamard = Hadamard(wires=0)
        s_gate = S(wires=0)
        s_dagger = S(wires=0, use_dagger=True)

        device.reset_states(1)
        hadamard(q_machine=device)
        s_gate(q_machine=device)
        print(device.states)

        device.reset_states(1)
        hadamard(q_machine=device)
        s_dagger(q_machine=device)
        print(device.states)

        # [[[0.7071068+0.j       ,0.       +0.j       ],
        #   [0.       +0.7071068j,0.       +0.j       ]]]
        # <QTensor [1, 2, 2] DEV_CPU complex64>
        #
        # [[[0.7071068+0.j       ,0.       +0.j       ],
        #   [0.       -0.7071068j,0.       +0.j       ]]]
        # <QTensor [1, 2, 2] DEV_CPU complex64>


paulix
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.paulix(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 paulix 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import paulix,QMachine
        qm  = QMachine(4)
        paulix(q_machine=qm, wires=1,)
        print(qm.states)

        # [[[[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

PauliX
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.PauliX(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个PauliX逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 PauliX逻辑门类实例。

    Example::
        
        from pyvqnet.qnn.vqc import PauliX,QMachine
        device = QMachine(4)
        layer = PauliX(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)

pauliy
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.pauliy(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 pauliy 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import pauliy,QMachine
        qm  = QMachine(4)
        pauliy(q_machine=qm, wires=1,)
        print(qm.states)

        # [[[[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+1.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

PauliY
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.PauliY(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个PauliY逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 PauliY逻辑门类实例。

    Example::
        
        from pyvqnet.qnn.vqc import PauliY,QMachine
        device = QMachine(4)
        layer = PauliY(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)

pauliz
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.pauliz(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 pauliz 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import pauliz,QMachine
        qm  = QMachine(4)
        pauliz(q_machine=qm, wires=1,)
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

PauliZ
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.PauliZ(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个PauliZ逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 PauliZ逻辑门类实例。

    Example::
        
        from pyvqnet.qnn.vqc import PauliZ,QMachine
        device = QMachine(4)
        layer = PauliZ(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)

x1
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.x1(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 x1 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import x1,QMachine
        qm  = QMachine(4)
        x1(q_machine=qm, wires=1,)
        print(qm.states)

        # [[[[[0.7071068+0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       -0.7071068j 0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]
        # 
        # 
        #   [[[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]]]

X1
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.X1(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个X1逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 X1逻辑门类实例。

    Example::
        
        from pyvqnet.qnn.vqc import X1,QMachine
        device = QMachine(4)
        layer = X1(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)

rx
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.rx(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 rx 

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import rx,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        rx(q_machine=qm, wires=1,params=QTensor([0.5]))
        print(qm.states)

        # [[[[[0.9689124+0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       -0.247404j 0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]
        # 
        # 
        #   [[[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]]]

RX
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.RX(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RX逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RX逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import RX,QMachine
        device = QMachine(4)
        layer = RX(trainable=True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

ry
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.ry(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 ry 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import ry,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        ry(q_machine=qm, wires=1,params=QTensor([0.5]),)
        print(qm.states)

        # [[[[[0.9689124+0.j 0.       +0.j]
        #     [0.       +0.j 0.       +0.j]]
        # 
        #    [[0.247404 +0.j 0.       +0.j]
        #     [0.       +0.j 0.       +0.j]]]
        # 
        # 
        #   [[[0.       +0.j 0.       +0.j]
        #     [0.       +0.j 0.       +0.j]]
        # 
        #    [[0.       +0.j 0.       +0.j]
        #     [0.       +0.j 0.       +0.j]]]]]

RY
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.RY(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RY逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RY逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import RY,QMachine
        device = QMachine(4)
        layer = RY(trainable=True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

rz
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.rz(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 rz 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import rz,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        rz(q_machine=qm, wires=1,params=QTensor([0.5]),)
        print(qm.states)
        
        # [[[[[0.9689124-0.247404j 0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]
        # 
        # 
        #   [[[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]]]

RZ
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.RZ(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RZ逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RZ逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import RZ,QMachine
        device = QMachine(4)
        layer = RZ(trainable=True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

crx
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.crx(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 crx 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::

        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.crx(q_machine=qm,wires=[0,2], params=QTensor([0.5]),)
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]


CRX
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.CRX(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CRX逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CRX逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import CRX,QMachine
        device = QMachine(4)
        layer = CRX(trainable=True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

cry
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.cry(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 cry 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::

        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.cry(q_machine=qm,wires=[0,2], params=QTensor([0.5]))
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

CRY
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.CRY(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CRY逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CRY逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import CRY,QMachine
        device = QMachine(4)
        layer = CRY(trainable=True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

crz
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.crz(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 crz 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::

        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.crz(q_machine=qm,wires=[0,2], params=QTensor([0.5]))
        print(qm.states)
        
        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

CRZ
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.CRZ(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CRZ逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CRZ逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import CRZ,QMachine
        device = QMachine(4)
        layer = CRZ(trainable=True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



u1
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.u1(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 u1 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import u1,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        u1(q_machine=qm, wires=1,params=QTensor([24.0]),)
        print(qm.states)

        # [[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]

U1
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.U1(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个U1逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 U1逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import U1,QMachine
        device = QMachine(4)
        layer = U1(trainable=True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

u2
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.u2(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 u2 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import u2,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        u2(q_machine=qm, wires=1,params=QTensor([[24.0,-3]]),)
        print(qm.states)

        # [[[[[0.7071068+0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.2999398-0.6403406j 0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]
        # 
        # 
        #   [[[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]]]

U2
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.U2(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个U2逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 U2逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import U2,QMachine
        device = QMachine(4)
        layer = U2(trainable=True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

u3
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.u3(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 u3 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import u3,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        u3(q_machine=qm, wires=1,params=QTensor([[24.0,-3,1]]),)
        print(qm.states)

        # [[[[[0.843854 +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.5312032+0.0757212j 0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]
        # 
        # 
        #   [[[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]]]

U3
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.U3(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个U3逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 U3逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import U3,QMachine
        device = QMachine(4)
        layer = U3(trainable=True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

cy
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.cy(q_machine, wires, params=None, use_dagger=False)

    对q_machine中的态矢作用量子逻辑门 cy 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。

    Example::

        from pyvqnet.qnn.vqc import cy,QMachine
        qm = QMachine(4)
        cy(q_machine=qm,wires=(1,0))
        print(qm.states)
        # [[[[[1.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]],

        #    [[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]]],


        #   [[[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]],

        #    [[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]]]]]


CY
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.CY(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CY逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CY逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import CY,QMachine
        device = QMachine(4)
        layer = CY(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

cnot
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.cnot(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 cnot 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import cnot,QMachine
        qm  = QMachine(4)
        cnot(q_machine=qm,wires=[1,0],)
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

CNOT
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.CNOT(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CNOT逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CNOT逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import CNOT,QMachine
        device = QMachine(4)
        layer = CNOT(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

cr
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.cr(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 cr 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import cr,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        cr(q_machine=qm,wires=[1,0],params=QTensor([0.5]),)
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

CR
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.CR(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CR逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CR逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import CR,QMachine
        device = QMachine(4)
        layer = CR(trainable=True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

swap
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.swap(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 swap 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import swap,QMachine
        qm  = QMachine(4)
        swap(q_machine=qm,wires=[1,0],)
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]


SWAP
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.SWAP(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个SWAP逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 SWAP 逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import SWAP,QMachine
        device = QMachine(4)
        layer = SWAP(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


cswap
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.cswap(q_machine, wires, params=None, use_dagger=False)

    对q_machine中的态矢作用量子逻辑门 cswap 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::

        from pyvqnet.qnn.vqc import cswap,QMachine
        qm  = QMachine(4)
        cswap(q_machine=qm,wires=[1,0,3],)
        print(qm.states)
        # [[[[[1.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]],

        #    [[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]]],


        #   [[[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]],

        #    [[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]]]]]


CSWAP
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.CSWAP(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个SWAP逻辑门类 。

    .. math:: CSWAP = \begin{bmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
        \end{bmatrix}.

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CSWAP 逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import CSWAP,QMachine
        device = QMachine(4)
        layer = CSWAP(wires=[0,1,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)
        # [[[[[1.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]],

        #    [[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]]],


        #   [[[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]],

        #    [[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]]]],



        #  [[[[1.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]],

        #    [[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]]],


        #   [[[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]],

        #    [[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]]]]]


iswap
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.iswap(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 iswap 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::

        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.iswap(q_machine=qm,wires=[0,1])
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

cz
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.cz(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 cz 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import cz,QMachine
        qm  = QMachine(4)
        cz(q_machine=qm,wires=[1,0],)
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]


CZ
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.CZ(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CZ逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CZ 逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import CZ,QMachine
        device = QMachine(4)
        layer = CZ(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

rxx
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.rxx(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 rxx 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import rxx,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        rxx(q_machine=qm,wires=[1,0],params=QTensor([0.2]),)
        print(qm.states)

        # [[[[[0.9950042+0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]
        # 
        # 
        #   [[[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       -0.0998334j 0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]]]

RXX
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.RXX(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RXX逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个RXX逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import RXX,QMachine
        device = QMachine(4)
        layer = RXX(trainable=True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

ryy
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.ryy(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 ryy 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import ryy,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        ryy(q_machine=qm,wires=[1,0],params=QTensor([0.2]),)
        print(qm.states)

        # [[[[[0.9950042+0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]
        # 
        # 
        #   [[[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.0998334j 0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]]]

RYY
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.RYY(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RYY逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RYY 逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import RYY,QMachine
        device = QMachine(4)
        layer = RYY(trainable=True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

rzz
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.rzz(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 rzz 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import rzz,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        rzz(q_machine=qm,wires=[1,0],params=QTensor([0.2]),)
        print(qm.states)

        # [[[[[0.9950042-0.0998334j 0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]
        # 
        # 
        #   [[[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]]]


RZZ
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.RZZ(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RZZ逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RZZ 逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import RZZ,QMachine
        device = QMachine(4)
        layer = RZZ(trainable=True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

rzx
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.rzx(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 RZX 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import rzx,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        rzx(q_machine=qm,wires=[1,0],params=QTensor([0.2]),)
        print(qm.states)

        # [[[[[0.9950042+0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]
        # 
        # 
        #   [[[0.       -0.0998334j 0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]]]

RZX
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.RZX(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RZX逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RZX 逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import RZX,QMachine
        device = QMachine(4)
        layer = RZX(trainable=True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

toffoli
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.toffoli(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 toffoli 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import toffoli,QMachine
        qm  = QMachine(4)
        toffoli(q_machine=qm,wires=[0,1,2],)
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]


Toffoli
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.Toffoli(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个Toffoli逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 Toffoli 逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import Toffoli,QMachine
        device = QMachine(4)
        layer = Toffoli(  wires=[0,2,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

isingxx
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.isingxx(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 isingxx 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.isingxx(q_machine=qm,wires=[0,1], params = QTensor([0.5]),)
        print(qm.states)

        # [[[[[0.9689124+0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]
        # 
        # 
        #   [[[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       -0.247404j 0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]]]


IsingXX
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.IsingXX(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个IsingXX逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个IsingXX 逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import IsingXX,QMachine
        device = QMachine(4)
        layer = IsingXX(trainable=True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

isingyy
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.isingyy(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 isingyy 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.isingyy(q_machine=qm,wires=[0,1], params = QTensor([0.5]),)
        print(qm.states)

        # [[[[[0.9689124+0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]
        # 
        # 
        #   [[[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       +0.247404j 0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]]]


IsingYY
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.IsingYY(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个IsingYY逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 IsingYY 逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import IsingYY,QMachine
        device = QMachine(4)
        layer = IsingYY(trainable=True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

isingzz
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.isingzz(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 isingzz 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.isingzz(q_machine=qm,wires=[0,1], params = QTensor([0.5]),)
        print(qm.states)

        # [[[[[0.9689124-0.247404j 0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]
        # 
        # 
        #   [[[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]]]

IsingZZ
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.IsingZZ(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个IsingZZ逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 IsingZZ 逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import IsingZZ,QMachine
        device = QMachine(4)
        layer = IsingZZ(trainable=True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

isingxy
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.isingxy(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 isingxy 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.isingxy(q_machine=qm,wires=[0,1], params = QTensor([0.5]),)
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

IsingXY
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.IsingXY(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个IsingXY逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 IsingXY 逻辑门类实例。
    
    Example::

        from pyvqnet.qnn.vqc import IsingXY,QMachine
        device = QMachine(4)
        layer = IsingXY(trainable=True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

phaseshift
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.phaseshift(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 phaseshift 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.phaseshift(q_machine=qm,wires=[0], params = QTensor([0.5]),)
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

PhaseShift
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.PhaseShift(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个PhaseShift逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 PhaseShift 逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import PhaseShift,QMachine
        device = QMachine(4)
        layer = PhaseShift(trainable=True, wires=1)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

multirz
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.multirz(q_machine, wires, params=None, use_dagger=False)

    对q_machine中的态矢作用量子逻辑门 multirz :math:`\exp\left(-i\frac{\theta}{2} Z\otimes \cdots \otimes Z\right)`,作用在 ``wires`` 上的2个或2个以上量子比特。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引,至少包含2个比特。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。

    Example::

        from pyvqnet.qnn.vqc import QMachine, hadamard, multirz
        from pyvqnet.tensor import QTensor

        qm = QMachine(3)
        for wire in range(3):
            hadamard(q_machine=qm, wires=wire)
        multirz(q_machine=qm, wires=[0, 1, 2], params=QTensor([0.5]))
        print(qm.states)

        # [[[[0.3425622-0.0874705j,0.3425622+0.0874705j],
        #    [0.3425622+0.0874705j,0.3425622-0.0874705j]],
        #
        #   [[0.3425622+0.0874705j,0.3425622-0.0874705j],
        #    [0.3425622-0.0874705j,0.3425622+0.0874705j]]]]
        # <QTensor [1, 2, 2, 2] DEV_CPU complex64>

MultiRZ
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.MultiRZ(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个MultiRZ逻辑门类 :math:`\exp\left(-i\frac{\theta}{2} Z\otimes \cdots \otimes Z\right)`,作用在 ``wires`` 上的2个或2个以上量子比特。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None,至少包含2个比特。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 MultiRZ 逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import QMachine, hadamard, MultiRZ
        from pyvqnet.utils import set_random_seed

        set_random_seed(42)
        device = QMachine(3)
        # 3 wires: the trainable angle is initialised by the seeded generator
        layer = MultiRZ(trainable=True, wires=[0, 1, 2])
        device.reset_states(1)
        for wire in range(3):
            hadamard(q_machine=device, wires=wire)
        layer(q_machine=device)
        print(layer.params)
        print(device.states)

        # [[5.3385983]]
        # <Parameter [1, 1] DEV_CPU float32>
        #
        # [[[[-0.3148489-0.160842j,-0.3148489+0.160842j],
        #    [-0.3148489+0.160842j,-0.3148489-0.160842j]],
        #
        #   [[-0.3148489+0.160842j,-0.3148489-0.160842j],
        #    [-0.3148489-0.160842j,-0.3148489+0.160842j]]]]
        # <QTensor [1, 2, 2, 2] DEV_CPU complex64>
        
sdg
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.sdg(q_machine, wires, params=None, use_dagger=False)

    对q_machine中的态矢作用量子逻辑门 :math:`S^\dagger`,即 ``s(q_machine, wires, use_dagger=True)``;``use_dagger=True`` 时作用的是 :math:`S`。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。

    Example::

        from pyvqnet.qnn.vqc import QMachine, hadamard, s, sdg

        qm = QMachine(2)
        hadamard(q_machine=qm, wires=0)
        sdg(q_machine=qm, wires=0)
        print(qm.states)

        qm = QMachine(2)
        hadamard(q_machine=qm, wires=0)
        s(q_machine=qm, wires=0, use_dagger=True)
        print(qm.states)

        # [[[0.7071068+0.j       ,0.       +0.j       ],
        #   [0.       -0.7071068j,0.       +0.j       ]]]
        # <QTensor [1, 2, 2] DEV_CPU complex64>
        #
        # [[[0.7071068+0.j       ,0.       +0.j       ],
        #   [0.       -0.7071068j,0.       +0.j       ]]]
        # <QTensor [1, 2, 2] DEV_CPU complex64>

tdg
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.tdg(q_machine, wires, params=None, use_dagger=False)

    对q_machine中的态矢作用量子逻辑门 :math:`T^\dagger`,即 ``t(q_machine, wires, use_dagger=True)``;``use_dagger=True`` 时作用的是 :math:`T`。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。

    Example::

        from pyvqnet.qnn.vqc import QMachine, hadamard, tdg

        qm = QMachine(2)
        hadamard(q_machine=qm, wires=0)
        tdg(q_machine=qm, wires=0)
        print(qm.states)

        qm = QMachine(2)
        hadamard(q_machine=qm, wires=0)
        tdg(q_machine=qm, wires=0, use_dagger=True)
        print(qm.states)

        # [[[0.7071068+0.j ,0.       +0.j ],
        #   [0.5      -0.5j,0.       +0.j ]]]
        # <QTensor [1, 2, 2] DEV_CPU complex64>
        #
        # [[[0.7071068+0.j ,0.       +0.j ],
        #   [0.5      +0.5j,0.       +0.j ]]]
        # <QTensor [1, 2, 2] DEV_CPU complex64>

controlledphaseshift
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.controlledphaseshift(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 controlledphaseshift 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        for i in range(4):
            vqc.hadamard(q_machine=qm, wires=i)
        vqc.controlledphaseshift(q_machine=qm,params=QTensor([0.5]),wires=[0,1],)
        print(qm.states)

        # [[[[[0.25     +0.j        0.25     +0.j       ]
        #     [0.25     +0.j        0.25     +0.j       ]]
        # 
        #    [[0.25     +0.j        0.25     +0.j       ]
        #     [0.25     +0.j        0.25     +0.j       ]]]
        # 
        # 
        #   [[[0.25     +0.j        0.25     +0.j       ]
        #     [0.25     +0.j        0.25     +0.j       ]]
        # 
        #    [[0.2193956+0.1198564j 0.2193956+0.1198564j]
        #     [0.2193956+0.1198564j 0.2193956+0.1198564j]]]]]


ControlledPhaseShift
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.ControlledPhaseShift(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个ControlledPhaseShift逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个ControlledPhaseShift。

    Example::

        from pyvqnet.qnn.vqc import ControlledPhaseShift,QMachine
        device = QMachine(4)
        layer = ControlledPhaseShift(trainable=True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



multicontrolledx
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.multicontrolledx(q_machine, wires, params=None, use_dagger=False,control_values=None)
    
    对q_machine中的态矢作用量子逻辑门 multicontrolledx 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    :param control_values: 控制值,默认为None,当比特位为1时控制。


    Example::
 


        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        for i in range(4):
            vqc.hadamard(q_machine=qm, wires=i)
        vqc.phaseshift(q_machine=qm,wires=[0], params = QTensor([0.5]))
        vqc.phaseshift(q_machine=qm,wires=[1], params = QTensor([2]))
        vqc.phaseshift(q_machine=qm,wires=[3], params = QTensor([3]))
        vqc.multicontrolledx(qm, wires=[0, 1, 3, 2])
        print(qm.states)

        # [[[[[ 0.25     +0.j       ,-0.2474981+0.03528j  ],
        #     [ 0.25     +0.j       ,-0.2474981+0.03528j  ]],

        #    [[-0.1040367+0.2273243j, 0.0709155-0.239731j ],
        #     [-0.1040367+0.2273243j, 0.0709155-0.239731j ]]],


        #   [[[ 0.2193956+0.1198564j,-0.2341141-0.0876958j],
        #     [ 0.2193956+0.1198564j,-0.2341141-0.0876958j]],

        #    [[-0.2002859+0.149618j , 0.1771674-0.176385j ],
        #     [-0.2002859+0.149618j , 0.1771674-0.176385j ]]]]]


MultiControlledX
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.MultiControlledX(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False,control_values=None)
    
    定义一个MultiControlledX逻辑门类 。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :param control_values: 控制值,默认为None,当比特位为1时控制。

    :return: 一个 MultiControlledX 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor,kcomplex64

        qm = QMachine(4,dtype=kcomplex64)
        qm.reset_states(2)
        for i in range(4):
            vqc.hadamard(q_machine=qm, wires=i)
        vqc.isingzz(q_machine=qm, params=QTensor([0.25]), wires=[1,0])
        vqc.double_excitation(q_machine=qm, params=QTensor([0.55]), wires=[0,1,2,3])

        mcx = vqc.MultiControlledX( 
                        init_params=None,
                        wires=[2,3,0,1],
                        dtype=kcomplex64,
                        use_dagger=False,control_values=[1,0,0])
        y = mcx(q_machine = qm)
        print(qm.states)
        """
        [[[[[0.2480494-0.0311687j,0.2480494-0.0311687j],
            [0.2480494+0.0311687j,0.1713719-0.0215338j]],

        [[0.2480494+0.0311687j,0.2480494+0.0311687j],
            [0.2480494-0.0311687j,0.2480494+0.0311687j]]],


        [[[0.2480494+0.0311687j,0.2480494+0.0311687j],
            [0.2480494+0.0311687j,0.2480494+0.0311687j]],

        [[0.306086 -0.0384613j,0.2480494-0.0311687j],
            [0.2480494-0.0311687j,0.2480494-0.0311687j]]]],



        [[[[0.2480494-0.0311687j,0.2480494-0.0311687j],
            [0.2480494+0.0311687j,0.1713719-0.0215338j]],

        [[0.2480494+0.0311687j,0.2480494+0.0311687j],
            [0.2480494-0.0311687j,0.2480494+0.0311687j]]],


        [[[0.2480494+0.0311687j,0.2480494+0.0311687j],
            [0.2480494+0.0311687j,0.2480494+0.0311687j]],

        [[0.306086 -0.0384613j,0.2480494-0.0311687j],
            [0.2480494-0.0311687j,0.2480494-0.0311687j]]]]]
        """

single_excitation
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.single_excitation(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 single_excitation 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.single_excitation(q_machine=qm, wires=[0, 1], params=QTensor([0.5]))
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

double_excitation
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.double_excitation(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 double_excitation 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        for i in range(4):
            vqc.hadamard(q_machine=qm, wires=i)
        vqc.isingzz(q_machine=qm, params=QTensor([0.55]), wires=[1,0])
        vqc.double_excitation(q_machine=qm, params=QTensor([0.55]), wires=[0,1,2,3])
        print(qm.states)

        # [[[[[0.2406063-0.0678867j 0.2406063-0.0678867j]
        #     [0.2406063-0.0678867j 0.1662296-0.0469015j]]
        # 
        #    [[0.2406063+0.0678867j 0.2406063+0.0678867j]
        #     [0.2406063+0.0678867j 0.2406063+0.0678867j]]]
        # 
        # 
        #   [[[0.2406063+0.0678867j 0.2406063+0.0678867j]
        #     [0.2406063+0.0678867j 0.2406063+0.0678867j]]
        # 
        #    [[0.2969014-0.0837703j 0.2406063-0.0678867j]
        #     [0.2406063-0.0678867j 0.2406063-0.0678867j]]]]]  



unitary
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.unitary(q_machine, wires, mat, use_dagger=False)

    对q_machine中的态矢作用任意酉矩阵 ``mat`` 表示的逻辑门。``mat`` 为常量矩阵,前面逻辑门参数的梯度可以穿过该门继续反向传播。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引,``mat`` 作用的比特。
    :param mat: 酉矩阵,``QTensor``,形状为 ``(2**len(wires), 2**len(wires))`` 的方阵,行、列按照 ``wires`` 中比特上的计算基矢排列(第一个比特为最高位);作用时转换为模拟器的数据类型。
    :param use_dagger: 是否共轭转置,默认为False。

    Example::

        import numpy as np
        from pyvqnet import tensor
        from pyvqnet.qnn.vqc import QMachine, hadamard, unitary, Probability

        T = tensor.QTensor(np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex64))
        CNOT = tensor.QTensor(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex64))

        qm = QMachine(2)
        hadamard(q_machine=qm, wires=0)
        unitary(q_machine=qm, wires=0, mat=T)
        hadamard(q_machine=qm, wires=0)
        print(Probability(wires=[0])(q_machine=qm))

        qm = QMachine(2)
        hadamard(q_machine=qm, wires=0)
        unitary(q_machine=qm, wires=0, mat=T)
        unitary(q_machine=qm, wires=0, mat=T, use_dagger=True)
        hadamard(q_machine=qm, wires=0)
        print(Probability(wires=[0])(q_machine=qm))

        qm = QMachine(2)
        hadamard(q_machine=qm, wires=0)
        unitary(q_machine=qm, wires=[0, 1], mat=CNOT)
        print(Probability(wires=[0, 1])(q_machine=qm))

        # [[0.8535532,0.1464466]]
        # <QTensor [1, 2] DEV_CPU float32>
        #
        # [[0.9999998,0.       ]]
        # <QTensor [1, 2] DEV_CPU float32>
        #
        # [[0.5,0. ,0. ,0.5]]
        # <QTensor [1, 4] DEV_CPU float32>


Unitary
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.Unitary(matrix=None, wires=None, dtype=None, use_dagger=False, **kwargs)

    定义一个任意酉矩阵逻辑门类 。

    :param matrix: 作用的酉矩阵,``QTensor`` 或者 ``numpy.ndarray``,形状为 ``(2**len(wires), 2**len(wires))`` 的方阵,行、列按照 ``wires`` 中比特上的计算基矢排列(第一个比特为最高位)。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,默认为None,对应pyvqnet.kcomplex64;``matrix`` 在作用时转换为模拟器的数据类型。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 Unitary 逻辑门类实例。

    Example::

        import numpy as np
        from pyvqnet import tensor
        from pyvqnet.qnn.vqc import QMachine, Hadamard, Unitary, Probability

        T = tensor.QTensor(np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex64))
        device = QMachine(2)
        hadamard = Hadamard(wires=0)
        t_gate = Unitary(matrix=T, wires=0)
        t_dagger = Unitary(matrix=T, wires=0, use_dagger=True)

        device.reset_states(1)
        hadamard(q_machine=device)
        t_gate(q_machine=device)
        hadamard(q_machine=device)
        print(Probability(wires=[0])(q_machine=device))

        device.reset_states(1)
        hadamard(q_machine=device)
        t_gate(q_machine=device)
        t_dagger(q_machine=device)
        hadamard(q_machine=device)
        print(Probability(wires=[0])(q_machine=device))

        # [[0.8535532,0.1464466]]
        # <QTensor [1, 2] DEV_CPU float32>
        #
        # [[0.9999998,0.       ]]
        # <QTensor [1, 2] DEV_CPU float32>

测量接口
===================


VQC_Purity
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_Purity(q_machine, qubits_idx, num_wires=None)

    计算量子态在量子比特 ``qubits_idx`` 上的子系统的纯度。

    .. math::
        \gamma = \text{Tr}(\rho^2)

    式中 :math:`\rho` 为 ``qubits_idx`` 上的约化密度矩阵。标准化量子态的纯度满足 :math:`\frac{1}{d} \leq \gamma \leq 1` ,
    其中 :math:`d` 是该子系统的希尔伯特空间的维数。
    纯态的纯度是1。

    :param q_machine: ``pyvqnet.qnn.vqc.QMachine`` 模拟器;也可以直接传入量子态 ``QTensor``,此时 ``num_wires`` 为必填,量子态的形状为 ``(batchsize, 2, ..., 2)`` 或 ``(batchsize, 2**num_wires)``。
    :param qubits_idx: 要计算纯度的量子比特位索引,整数或者列表。
    :param num_wires: 量子比特数,默认值:None;传入模拟器时由模拟器给出。

    :return: 对应比特位置上的纯度,``QTensor``。批处理数量为1时形状为 ``(1, 1)``,否则为 ``(batchsize,)``,其中 batchsize 为 ``q_machine.reset_states(b)`` 的批处理数量b;``qubits_idx`` 覆盖全部比特时直接返回整数 ``1``。

    Example::

        from pyvqnet.qnn.vqc import VQC_Purity, vqc_purity, rx, ry, cnot, QMachine
        from pyvqnet.tensor import QTensor

        x = QTensor([[0.7, 0.4], [1.7, 2.4]], requires_grad=True)
        qm = QMachine(3)
        qm.reset_states(2)
        rx(q_machine=qm, wires=0, params=x[:, [0]])
        ry(q_machine=qm, wires=1, params=x[:, [1]])
        ry(q_machine=qm, wires=2, params=x[:, [1]])
        cnot(q_machine=qm, wires=[0, 1])
        cnot(q_machine=qm, wires=[2, 1])
        y = VQC_Purity(qm, [0, 1])
        y.backward()
        print(y)
        print(x.grad)

        # the raw-state form needs the wire count
        print(vqc_purity(qm.states, [0, 1], num_wires=3))
        print(VQC_Purity(qm, 2))

        # [0.9356752,0.8759571]
        # <QTensor [2] DEV_CPU float32>
        #
        # [[-0.0000001,-0.2498935],
        #  [ 0.0000001, 0.0435818]]
        # <QTensor [2, 2] DEV_CPU float32>
        #
        # [0.9356752,0.8759571]
        # <QTensor [2] DEV_CPU float32>
        #
        # [0.9356751,0.8759571]
        # <QTensor [2] DEV_CPU float32>


vqc_purity
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.vqc_purity(q_machine, qubits_idx, num_wires=None)

    与 ``pyvqnet.qnn.vqc.VQC_Purity`` 是同一个函数(``VQC_Purity`` 是它的大写别名),参数和返回值相同。

VQC_VarMeasure
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_VarMeasure(q_machine, obs)

    计算可观测量 ``obs`` 在 ``q_machine`` 当前量子态上的方差 :math:`\langle H^2\rangle - \langle H\rangle^2`。

    :param q_machine: ``pyvqnet.qnn.vqc.QMachine`` 模拟器。
    :param obs: 测量观测量,当前支持Hadamard,I,PauliX,PauliY,PauliZ 几种Observable.

    :return: 计算可观测量方差,``QTensor``,形状为 ``(batchsize, 1)``,其中 batchsize 为 ``q_machine.reset_states(b)`` 的批处理数量b。计算中对角化 ``obs`` 所作用的逻辑门在返回前会被撤销,``q_machine`` 的态矢保持不变。

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc import VQC_VarMeasure, MeasureAll, Probability, rx, hadamard, QMachine, PauliY

        x = QTensor([[0.5]], requires_grad=True)
        qm = QMachine(3)
        rx(q_machine=qm, wires=0, params=x)
        hadamard(q_machine=qm, wires=1)
        var_result = VQC_VarMeasure(q_machine=qm, obs=PauliY(wires=0))
        var_result.backward()
        print(var_result)
        print(x.grad)

        # the machine's state is left unchanged by the variance call
        print(Probability(wires=[0, 1])(q_machine=qm))
        print(MeasureAll(obs={"Y0": 1.0})(q_machine=qm))

        # [[0.770151]]
        # <QTensor [1, 1] DEV_CPU float32>
        #
        # [[-0.8414709]]
        # <QTensor [1, 1] DEV_CPU float32>
        #
        # [[0.4693956,0.4693956,0.0306044,0.0306044]]
        # <QTensor [1, 4] DEV_CPU float32>
        #
        # [[-0.4794255]]
        # <QTensor [1, 1] DEV_CPU float32>


VQC_DensityMatrixFromQstate
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_DensityMatrixFromQstate(state, indices, num_wires=None)

    计算量子态在一组特定量子比特 ``indices`` 上的密度矩阵。

    :param state: 量子态,``QTensor``。未给出 ``num_wires`` 时为态矢,形状 ``(2**N,)``(N 为量子比特个数,一维状态向量,基矢按 000 -> 111 的顺序)、``(batchsize, 2**N)`` 或 ``(batchsize, 2, ..., 2)``;给出 ``num_wires`` 时为密度矩阵,形状需要是 ``(2**N, 2**N)`` 或 ``(batchsize, 2**N, 2**N)``,此时返回它在 ``indices`` 上的约化密度矩阵。
    :param indices: 所考虑子系统中的量子比特索引列表。
    :param num_wires: 量子比特数,默认值:None;仅在 ``state`` 为密度矩阵时需要。
    :return: 密度矩阵,``QTensor``,数据类型与 ``state`` 相同。带 batch 维度的输入返回大小为 ``(b, 2**len(indices), 2**len(indices))`` 的密度矩阵,其中b为 ``q_machine.reset_states(b)`` 的批处理数量b;形状为 ``(2**N,)`` 的输入返回 ``(2**len(indices), 2**len(indices))``。

    Example::

        from pyvqnet.qnn.vqc import VQC_DensityMatrixFromQstate, rx, ry, cnot, QMachine
        from pyvqnet.tensor import QTensor

        x = QTensor([[0.7, 0.4], [1.7, 2.4]], requires_grad=True)
        qm = QMachine(3)
        qm.reset_states(2)
        rx(q_machine=qm, wires=0, params=x[:, [0]])
        ry(q_machine=qm, wires=1, params=x[:, [1]])
        ry(q_machine=qm, wires=2, params=x[:, [1]])
        cnot(q_machine=qm, wires=[0, 1])
        cnot(q_machine=qm, wires=[2, 1])
        y = VQC_DensityMatrixFromQstate(qm.states, [0, 1])
        print(y)
        print(y.shape, y.dtype)

        # a density-matrix input needs num_wires
        rho = VQC_DensityMatrixFromQstate(qm.states, [0, 1, 2])
        print(rho.shape)
        print(VQC_DensityMatrixFromQstate(rho, [0], num_wires=3))

        # [[[0.8155131+0.j       ,0.1718155+0.j       ,0.       +0.0627175j,
        #    0.       +0.2976855j],
        #   [0.1718155+0.j       ,0.0669081+0.j       ,0.       +0.0244234j,
        #    0.       +0.0627175j],
        #   [0.       -0.0627175j,0.       -0.0244234j,0.0089152+0.j       ,
        #    0.0228937+0.j       ],
        #   [0.       -0.2976855j,0.       -0.0627175j,0.0228937+0.j       ,
        #    0.1086637+0.j       ]],
        #
        #  [[0.3362115+0.j       ,0.1471084+0.j       ,0.       +0.1674583j,
        #    0.       +0.3827206j],
        #   [0.1471084+0.j       ,0.0993663+0.j       ,0.       +0.1131119j,
        #    0.       +0.1674583j],
        #   [0.       -0.1674583j,0.       -0.1131119j,0.1287589+0.j       ,
        #    0.1906232+0.j       ],
        #   [0.       -0.3827206j,0.       -0.1674583j,0.1906232+0.j       ,
        #    0.4356633+0.j       ]]]
        # <QTensor [2, 4, 4] DEV_CPU complex64>
        # [2, 4, 4] complex64
        # [2, 8, 8]
        #
        # [[[0.8824212+0.j       ,0.       +0.1254351j],
        #   [0.       -0.1254351j,0.1175789+0.j       ]],
        #
        #  [[0.4355778+0.j       ,0.       +0.3349165j],
        #   [0.       -0.3349165j,0.5644222+0.j       ]]]
        # <QTensor [2, 2, 2] DEV_CPU complex64>



VQC_DensityMatrix
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_DensityMatrix(q_machine, indices)

    计算模拟器当前量子态在量子比特 ``indices`` 上的密度矩阵。小写函数名 ``pyvqnet.qnn.vqc.common.qmeasure.qinfo.vqc_density_matrix`` 是同一个函数。

    :param q_machine: ``pyvqnet.qnn.vqc.QMachine`` 模拟器。
    :param indices: 所考虑子系统中的量子比特索引列表。
    :return: 密度矩阵,``QTensor``,形状为 ``(b, 2**len(indices), 2**len(indices))``,其中b为 ``q_machine.reset_states(b)`` 的批处理数量b。

    Example::

        from pyvqnet.qnn.vqc import VQC_DensityMatrix, hadamard, cnot, QMachine

        qm = QMachine(2)
        hadamard(q_machine=qm, wires=0)
        cnot(q_machine=qm, wires=[0, 1])
        print(VQC_DensityMatrix(qm, [0]))
        print(VQC_DensityMatrix(qm, [0, 1]))

        # [[[0.5+0.j,0. +0.j],
        #   [0. +0.j,0.5+0.j]]]
        # <QTensor [1, 2, 2] DEV_CPU complex64>
        #
        # [[[0.5+0.j,0. +0.j,0. +0.j,0.5+0.j],
        #   [0. +0.j,0. +0.j,0. +0.j,0. +0.j],
        #   [0. +0.j,0. +0.j,0. +0.j,0. +0.j],
        #   [0.5+0.j,0. +0.j,0. +0.j,0.5+0.j]]]
        # <QTensor [1, 4, 4] DEV_CPU complex64>


VQC_PartialTrace
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_PartialTrace(q_machine, indices, num_wires=None)

    计算对量子比特 ``indices`` 求偏迹(即丢弃这些比特)之后,剩余子系统上的密度矩阵。小写函数名 ``pyvqnet.qnn.vqc.common.qmeasure.qinfo.vqc_partial_trace`` 是同一个函数。

    :param q_machine: ``pyvqnet.qnn.vqc.QMachine`` 模拟器;也可以是态矢 ``QTensor``,形状为 ``(2**N,)``、``(batchsize, 2**N)`` 或 ``(batchsize, 2, ..., 2)``,或者给出了 ``num_wires`` 的密度矩阵张量。
    :param indices: 被求迹(丢弃)的量子比特索引列表。
    :param num_wires: 量子比特数,默认值:None;仅在 ``q_machine`` 为密度矩阵张量(形状 ``(2**N, 2**N)`` 或 ``(batchsize, 2**N, 2**N)``)时给出。
    :return: 剩余子系统上的密度矩阵,``QTensor``,形状为 ``(b, 1, 2**m, 2**m)``,其中 :math:`m` 为剩余的量子比特数,b 为批处理数量;一维态矢输入返回 ``(1, 2**m, 2**m)``。

    Example::

        from pyvqnet.qnn.vqc import VQC_PartialTrace, hadamard, cnot, QMachine

        qm = QMachine(3)
        hadamard(q_machine=qm, wires=0)
        cnot(q_machine=qm, wires=[0, 1])
        # trace out wire 2, keep wires 0 and 1
        y = VQC_PartialTrace(qm, [2])
        print(y)
        print(y.shape)

        # [[[[0.5+0.j,0. +0.j,0. +0.j,0.5+0.j],
        #    [0. +0.j,0. +0.j,0. +0.j,0. +0.j],
        #    [0. +0.j,0. +0.j,0. +0.j,0. +0.j],
        #    [0.5+0.j,0. +0.j,0. +0.j,0.5+0.j]]]]
        # <QTensor [1, 1, 4, 4] DEV_CPU complex64>
        # [1, 1, 4, 4]


VQC_VN_Entropy
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_VN_Entropy(q_machine, indices, base=None)

    计算量子态在量子比特 ``indices`` 上的子系统的 von Neumann 熵 :math:`S = -\text{Tr}(\rho \ln \rho)`,其中 :math:`\rho` 为该子系统上的约化密度矩阵,小于 ``1e-8`` 的本征值按0计算。小写函数名 ``pyvqnet.qnn.vqc.common.qmeasure.qinfo.vqc_vn_entropy`` 是同一个函数。

    :param q_machine: ``pyvqnet.qnn.vqc.QMachine`` 模拟器。
    :param indices: 所考虑子系统中的量子比特索引列表。
    :param base: 对数的底,默认值:None,使用自然对数。
    :return: 每个样本的熵,``QTensor``,形状为 ``(b,)``,其中b为 ``q_machine.reset_states(b)`` 的批处理数量b。

    Example::

        from pyvqnet.qnn.vqc import VQC_VN_Entropy, hadamard, cnot, ry, QMachine
        from pyvqnet.tensor import QTensor

        qm = QMachine(2)
        hadamard(q_machine=qm, wires=0)
        cnot(q_machine=qm, wires=[0, 1])
        print(VQC_VN_Entropy(qm, [0]))
        print(VQC_VN_Entropy(qm, [0], base=2))
        print(VQC_VN_Entropy(qm, [0, 1]))

        qm = QMachine(1)
        ry(q_machine=qm, wires=0, params=QTensor([0.5]))
        print(VQC_VN_Entropy(qm, [0]))

        # [0.6931472]
        # <QTensor [1] DEV_CPU float32>
        #
        # [1.]
        # <QTensor [1] DEV_CPU float32>
        #
        # [0.0000001]
        # <QTensor [1] DEV_CPU float32>
        #
        # [0.0000001]
        # <QTensor [1] DEV_CPU float32>


VQC_Mutual_Info
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_Mutual_Info(q_machine, indices0, indices1, base=None)

    计算两个子系统 ``indices0`` 和 ``indices1`` 之间的互信息 :math:`I(A:B) = S(\rho_A) + S(\rho_B) - S(\rho_{AB})`,其中 :math:`S` 为 von Neumann 熵。小写函数名 ``pyvqnet.qnn.vqc.common.qmeasure.qinfo.vqc_mutual_info`` 是同一个函数。

    :param q_machine: ``pyvqnet.qnn.vqc.QMachine`` 模拟器。
    :param indices0: 子系统 A 的量子比特索引列表。
    :param indices1: 子系统 B 的量子比特索引列表,两个子系统不能包含相同的量子比特,否则报 ``ValueError``。
    :param base: 对数的底,默认值:None,使用自然对数。
    :return: 每个样本的互信息,``QTensor``,形状为 ``(b,)``,其中b为 ``q_machine.reset_states(b)`` 的批处理数量b。

    Example::

        from pyvqnet.qnn.vqc import VQC_Mutual_Info, hadamard, cnot, QMachine

        qm = QMachine(2)
        hadamard(q_machine=qm, wires=0)
        cnot(q_machine=qm, wires=[0, 1])
        print(VQC_Mutual_Info(qm, [0], [1]))
        print(VQC_Mutual_Info(qm, [0], [1], base=2))

        # [1.3862942]
        # <QTensor [1] DEV_CPU float32>
        #
        # [1.9999999]
        # <QTensor [1] DEV_CPU float32>


VQC_MeyerWallachMeasure
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_MeyerWallachMeasure(q_machine, num_wires=None)

    计算量子态的 Meyer-Wallach 纠缠度 :math:`Q = 2\left(1 - \frac{1}{n}\sum_{k=0}^{n-1} \text{Tr}(\rho_k^2)\right)`,其中 :math:`\rho_k` 为第 :math:`k` 个量子比特上的约化密度矩阵,:math:`n` 为量子比特数;乘积态的取值为0,最大纠缠态的取值为1。小写函数名 ``pyvqnet.qnn.vqc.common.qmeasure.qinfo.vqc_meyer_wallach_measure`` 是同一个函数。

    :param q_machine: ``pyvqnet.qnn.vqc.QMachine`` 模拟器;也可以是给出了 ``num_wires`` 的密度矩阵张量。
    :param num_wires: 量子比特数,默认值:None;传入模拟器时不需要,仅在 ``q_machine`` 为密度矩阵张量时给出。
    :return: 每个样本的 Meyer-Wallach 纠缠度,``QTensor``,形状为 ``(b,)``,其中b为 ``q_machine.reset_states(b)`` 的批处理数量b。

    Example::

        from pyvqnet.qnn.vqc import VQC_MeyerWallachMeasure, hadamard, cnot, ry, QMachine
        from pyvqnet.tensor import QTensor

        qm = QMachine(3)
        hadamard(q_machine=qm, wires=0)
        cnot(q_machine=qm, wires=[0, 1])
        ry(q_machine=qm, wires=2, params=QTensor([1.0]))
        print(VQC_MeyerWallachMeasure(qm))

        # [0.6666667]
        # <QTensor [1] DEV_CPU float32>

Probability
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.Probability(wires, name="")

    一个计算量子线路在特定比特上概率测量的Module类。

    :param wires: 测量比特的索引,列表、元组或者整数。
    :param name: 模块的名字,默认:""。
    :return: 一个概率测量的类。

    .. py:method:: forward(q_machine)

        进行概率测量计算

        :param q_machine: 作用的量子态矢模拟器
        :return: 概率测量结果

    .. note::

        使用该类进行计算的概率测量结果一般为 ``[b, 2**len(wires)]``,其中b为 ``q_machine.reset_states(b)`` 的批处理数量b,``wires`` 为单个整数时 ``len(wires)`` 为1。测量部分比特时,结果按照 ``wires`` 列表顺序对应的计算基矢排列(列表中第一个比特为最高位);测量全部比特时,结果按照比特索引升序排列。

        结果的数据类型与模拟器的精度对应:``pyvqnet.kcomplex64`` 的态为 ``pyvqnet.kfloat32``,``pyvqnet.kcomplex128`` 的态为 ``pyvqnet.kfloat64``,``state_storage=pyvqnet.kcomplex32`` 的态为 ``pyvqnet.kfloat16``。该测量可以在 ``pyvqnet.qnn.vqc.QuantumLayerAdjoint`` 中使用,结果可以对线路参数求梯度。

    Example::

        from pyvqnet.qnn.vqc import Probability, rx, ry, cnot, QMachine, rz
        from pyvqnet.tensor import QTensor

        x = QTensor([[0.56, 0.1], [0.56, 0.1]], requires_grad=True)
        qm = QMachine(4)
        qm.reset_states(2)
        rz(q_machine=qm, wires=0, params=x[:, [0]])
        rz(q_machine=qm, wires=1, params=x[:, [0]])
        cnot(q_machine=qm, wires=[0, 1])
        ry(q_machine=qm, wires=2, params=x[:, [1]])
        cnot(q_machine=qm, wires=[0, 2])
        rz(q_machine=qm, wires=3, params=x[:, [1]])
        ma = Probability(wires=1)
        y = ma(q_machine=qm)
        print(y)

        # [[1.0000004,0.       ],
        #  [1.0000004,0.       ]]
        # <QTensor [2, 2] DEV_CPU float32>

    对概率测量结果求梯度:

    .. code-block::

        from pyvqnet.qnn.vqc import Probability, QMachine, ry
        from pyvqnet import tensor

        x = tensor.QTensor([[0.5]], requires_grad=True)
        qm = QMachine(2)
        qm.reset_states(1)
        ry(q_machine=qm, wires=0, params=x)
        y = Probability(wires=[0])(q_machine=qm)
        print(y, y.dtype)
        y[:, 1].backward()
        print(x.grad)

        # [[0.9387913,0.0612087]]
        # <QTensor [1, 2] DEV_CPU float32> float32
        #
        # [[0.2397128]]
        # <QTensor [1, 1] DEV_CPU float32>


MeasureAll
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.MeasureAll(obs, name="")

    计算量子线路的测量结果,支持输入观测量 ``obs``。其格式可以为字典格式，用于表示一个由多个Pauli算符组合而成的可观测量；列表形式,表示多个期望值的可观测量列表。

    例如:

    {\'X0\': 0.23} 表示在量子比特0上作用PauliX,系数为0.23

    {\'X1 Z2\':2.4,\'Y2\':-0.5} 对应于观测量 2.4 * X1 @ Z2 - 0.5 * Y2

    [{\'X1 Z2\':4,\'Z1 Z0\':3},{\'X1 Y2 Z0\':3.5}] 对应于两个观测量 4 * X1 @ Z2 + 3 * Z1 @ Z0 以及 3.5 * X1 @ Y2 @ Z0 。

    :param obs: 测量观测量。
    :param name: 模块的名字,默认:""。

    .. py:method:: forward(q_machine)

        进行测量操作

        :param q_machine: 作用的量子态矢模拟器
        :return: 测量结果,QTensor。

    .. note::

        如果 ``obs`` 是列表,使用该类进行计算的测量结果一般为 ``[b, obs的列表长度]``,其中b为 ``q_machine.reset_states(b)`` 的批处理数量b。
        如果 ``obs`` 是字典,使用该类进行计算的测量结果一般为 ``[b, 1]``,其中b为 ``q_machine.reset_states(b)`` 的批处理数量b。

        结果的数据类型与模拟器的精度对应:``pyvqnet.kcomplex64`` 的态为 ``pyvqnet.kfloat32``,``pyvqnet.kcomplex128`` 的态为 ``pyvqnet.kfloat64``,``state_storage=pyvqnet.kcomplex32`` 的态为 ``pyvqnet.kfloat16``。该测量可以在 ``pyvqnet.qnn.vqc.QuantumLayerAdjoint`` 中使用。

    Example::

        from pyvqnet.qnn.vqc import MeasureAll, rx, ry, cnot, QMachine
        from pyvqnet.tensor import QTensor

        x = QTensor([[0.3, 0.5, 1.2], [0.9, -0.4, 0.1]], requires_grad=True)

        # 3 qubits
        qm = QMachine(3)
        # batch size = 2
        qm.reset_states(2)

        ry(q_machine=qm, wires=0, params=x[:, [0]])
        ry(q_machine=qm, wires=1, params=x[:, [1]])
        cnot(q_machine=qm, wires=[0, 1])
        rx(q_machine=qm, wires=2, params=x[:, [2]])

        # list of 3 observables: one output column per list entry
        obs_list = [{"Z0": 1.0}, {"X1": 1.0}, {"Y2": 2.0}]

        ma = MeasureAll(obs=obs_list)
        y = ma(q_machine=qm)
        print(y)

        # dict: one Pauli sum, one output column
        ma = MeasureAll(obs={"Z0": 1.0, "X1": -0.5})
        y = ma(q_machine=qm)
        print(y)
        y.backward()
        print(x.grad)

        # [[ 0.9553365, 0.4794255,-1.8640782],
        #  [ 0.62161  ,-0.3894184,-0.1996669]]
        # <QTensor [2, 3] DEV_CPU float32>
        #
        # [[0.7156237],
        #  [0.8163192]]
        # <QTensor [2, 1] DEV_CPU float32>
        #
        # [[-0.2955202,-0.4387913, 0.       ],
        #  [-0.783327 ,-0.4605306, 0.       ]]
        # <QTensor [2, 3] DEV_CPU float32>
 


Samples
----------------------------

.. py:class:: pyvqnet.qnn.vqc.Samples(wires=None, obs=None, shots = 1,name="")

    获取特定线路 ``wires`` 上的带有 ``shots`` 的观测量 ``obs`` 结果。

    :param wires: 样本量子比特索引,整数或者列表。默认值:None,根据运行时使用模拟器的所有比特。
    :param obs: 该值只能设为None。
    :param shots: 样本重复次数,默认值:1。
    :param name: 此模块的名称,默认值:“”。
    :return: 一个测量方法类

    .. py:method:: forward(q_machine)

        进行采样操作。

        :param q_machine: 作用的量子态矢模拟器
        :return: 测量结果,QTensor。

    .. note::

        使用该类进行计算的测量结果一般为 ``[b, shots, len(wires)]`` 的 ``pyvqnet.kint64`` 张量,其中b为 ``q_machine.reset_states(b)`` 的批处理数量b,每个元素是对应比特上的一次采样结果(0或1),各列按照 ``wires`` 的顺序排列。

        样本从计算基矢上的概率分布中抽取,随机数来自当前后端的默认生成器,可以用 ``pyvqnet.utils.set_random_seed`` 固定:同一种子下重复调用会依次推进随机序列。采样结果是整数张量,不能对其求梯度;该测量不能在 ``pyvqnet.qnn.vqc.QuantumLayerAdjoint`` 中使用。

    Example::

        from pyvqnet.qnn.vqc import Samples, rx, ry, cnot, QMachine, rz
        from pyvqnet.tensor import QTensor
        from pyvqnet.utils import set_random_seed

        x = QTensor([[0.56, 0.1], [0.56, 0.1]], requires_grad=True)

        set_random_seed(42)
        qm = QMachine(4)
        qm.reset_states(2)
        rz(q_machine=qm, wires=0, params=x[:, [0]])
        rx(q_machine=qm, wires=1, params=x[:, [0]])
        cnot(q_machine=qm, wires=[0, 1])

        cnot(q_machine=qm, wires=[0, 2])
        ry(q_machine=qm, wires=3, params=x[:, [1]])

        ma = Samples(wires=[0, 1, 2], shots=3)
        y = ma(q_machine=qm)
        print(y)

        # [[[0,0,0],
        #   [0,0,0],
        #   [0,0,0]],
        #
        #  [[0,1,0],
        #   [0,0,0],
        #   [0,0,0]]]
        # <QTensor [2, 3, 3] DEV_CPU int64>



QInfoMeasurement
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.QInfoMeasurement(quantity=None, wires=None, obs=None, name="")

    把一个量子信息量作为线路的测量模块。``quantity`` 选择计算的量:

    * ``"purity"``: ``wires`` 上子系统的纯度 :math:`\text{Tr}(\rho^2)`。
    * ``"entropy"``: ``wires`` 上子系统的 von Neumann 熵 :math:`-\text{Tr}(\rho \ln \rho)`。
    * ``"mutual_info"``: 两个子系统之间的互信息 :math:`S(\rho_A) + S(\rho_B) - S(\rho_{AB})`。
    * ``"meyer_wallach"``: Meyer-Wallach 纠缠度 :math:`Q = 2\left(1 - \frac{1}{n}\sum_{k=0}^{n-1} \text{Tr}(\rho_k^2)\right)`,对模拟器的全部 :math:`n` 个量子比特计算。
    * ``"variance"``: 厄密矩阵 ``obs`` 的方差 :math:`\langle H^2\rangle - \langle H\rangle^2`。
    * ``"expectation"``: 厄密矩阵 ``obs`` 的期望 :math:`\langle H\rangle`。

    其中 :math:`\rho` 为对应子系统上的约化密度矩阵,:math:`S(\rho)` 为 von Neumann 熵,熵和互信息使用自然对数。

    :param quantity: 字符串,取值为 ``"purity"``、``"entropy"``、``"mutual_info"``、``"meyer_wallach"``、``"variance"`` 或者 ``"expectation"``,默认值:None。
    :param wires: 测量作用的量子比特索引。``"purity"`` 为整数或者列表;``"entropy"`` 为列表;``"mutual_info"`` 为两个比特索引列表,例如 ``[[0, 1], [2]]``,两个子系统不能包含相同的比特;``"meyer_wallach"`` 需要给出(内容不参与计算);``"variance"`` 和 ``"expectation"`` 为整数或者列表。``"variance"`` 和 ``"expectation"`` 省略 ``wires`` 时 ``obs`` 作用于全部量子比特。
    :param obs: ``"variance"`` 时为厄密矩阵,形状为 ``(2**len(wires), 2**len(wires))``,省略 ``wires`` 时为 ``(2**num_wires, 2**num_wires)``;``"expectation"`` 时为厄密矩阵,或者带有 ``observables``(或 ``matrix``)和可选 ``wires`` 的字典,例如 ``{"wires": (1, 0), "observables": 矩阵}``,矩阵形状需要与 ``wires`` 对应。省略 ``quantity`` 时,如果 ``obs`` 是矩阵或者含 ``observables``/``matrix`` 键的字典,按 ``"expectation"`` 计算。其他量不需要该参数。
    :param name: 模块的名字,默认:""。
    :return: 一个 QInfoMeasurement 测量方法类实例。

    .. py:method:: forward(q_machine)

        进行测量操作。

        :param q_machine: 作用的量子态矢模拟器
        :return: 测量结果,QTensor。

    .. note::

        返回结果中 b 为 ``q_machine.reset_states(b)`` 的批处理数量b。``"purity"``、``"variance"``、``"expectation"`` 的结果形状为 ``(b, 1)``,其中 ``"purity"`` 在 ``wires`` 覆盖全部量子比特时直接返回整数 ``1``;``"entropy"``、``"mutual_info"``、``"meyer_wallach"`` 的结果形状为 ``(b,)``。

        结果的数据类型与模拟器的精度对应:``pyvqnet.kcomplex64`` 的态为 ``pyvqnet.kfloat32``,``pyvqnet.kcomplex128`` 的态为 ``pyvqnet.kfloat64``,``state_storage=pyvqnet.kcomplex32`` 的态为 ``pyvqnet.kfloat16``(``"meyer_wallach"`` 仍为 ``pyvqnet.kfloat32``)。该测量可以在 ``pyvqnet.qnn.vqc.QuantumLayerAdjoint`` 中使用;其中 ``"expectation"`` 需要给出 ``wires``,并且 ``obs`` 的矩阵为整个寄存器大小 ``(2**num_wires, 2**num_wires)``。

    在张量网络后端的模拟器上(``pyvqnet.qnn.vqc.tn.native`` 和 ``pyvqnet.qnn.vqc.tn.torch``,MPS 与非 MPS 线路),只支持 ``"expectation"``(给出 ``wires`` 和 ``obs`` 矩阵,或者省略 ``wires`` 并给出整个寄存器大小的矩阵)、``"purity"`` 和 ``"variance"``,结果形状为 ``(b,)``;``"entropy"``、``"meyer_wallach"`` 和 ``"mutual_info"`` 报 ``NotImplementedError``。

    Example::

        import numpy as np
        from pyvqnet.qnn.vqc import QMachine, hadamard, cnot, ry, QInfoMeasurement
        from pyvqnet.tensor import QTensor

        X = np.array([[0, 1], [1, 0]], dtype=np.float64)

        # batch size 2: a Bell pair on wires 0, 1 and one rotation on wire 2
        qm = QMachine(3)
        qm.reset_states(2)
        hadamard(q_machine=qm, wires=0)
        cnot(q_machine=qm, wires=[0, 1])
        ry(q_machine=qm, wires=2, params=QTensor([[1.0], [0.5]]))

        purity = QInfoMeasurement("purity", wires=[0])(q_machine=qm)
        entropy = QInfoMeasurement("entropy", wires=[0])(q_machine=qm)
        mutual_info = QInfoMeasurement("mutual_info", wires=[[0], [1]])(q_machine=qm)
        meyer_wallach = QInfoMeasurement("meyer_wallach", wires=[0, 1, 2])(q_machine=qm)
        variance = QInfoMeasurement("variance", wires=[2], obs=X)(q_machine=qm)
        expectation = QInfoMeasurement("expectation", obs={"wires": (2,), "observables": X})(q_machine=qm)

        print(purity, purity.shape)
        print(entropy, entropy.shape)
        print(mutual_info, mutual_info.shape)
        print(meyer_wallach, meyer_wallach.shape)
        print(variance, variance.shape)
        print(expectation, expectation.shape)

        # [0.4999999,0.4999999]
        # <QTensor [2] DEV_CPU float32> [2]
        #
        # [0.6931472,0.6931472]
        # <QTensor [2] DEV_CPU float32> [2]
        #
        # [1.3862942,1.3862942]
        # <QTensor [2] DEV_CPU float32> [2]
        #
        # [0.6666667,0.6666667]
        # <QTensor [2] DEV_CPU float32> [2]
        #
        # [[0.2919267],
        #  [0.7701511]]
        # <QTensor [2, 1] DEV_CPU float32> [2, 1]
        #
        # [[0.8414709],
        #  [0.4794255]]
        # <QTensor [2, 1] DEV_CPU float32> [2, 1]

    省略 ``quantity`` 并以字典形式给出 ``obs``,计算厄密观测量的期望:

    Example::

        from pyvqnet.qnn.vqc import qcircuit
        from pyvqnet.qnn.vqc import QMachine, RX, RY, CNOT, PauliX, PauliZ, VQC_RotCircuit, QInfoMeasurement
        from pyvqnet.tensor import QTensor, tensor
        import pyvqnet
        from pyvqnet.nn import Parameter
        import numpy as np
        bsz = 3
        H = np.array([[8, 4, 0, -6], [4, 0, 4, 0], [0, 4, 8, 0], [-6, 0, 0, 0]])
        class QModel(pyvqnet.nn.Module):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()
                self.rot_param = Parameter((3, ))
                self.rot_param.copy_value_from(tensor.QTensor([-0.5, 1, 2.3]))
                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype)
                self.rx_layer1 = VQC_RotCircuit
                self.ry_layer2 = RY(trainable=True,
                                    wires=0,
                                    init_params=tensor.QTensor([-0.5]))
                self.xlayer = PauliX(wires=0)
                self.cnot = CNOT(wires=[0, 1])
                self.measure = QInfoMeasurement(obs={'wires': (1, 0), 'observables': tensor.to_tensor(H)})

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])

                qcircuit.rx(q_machine=self.qm, wires=0, params=x[:, [1]])
                qcircuit.ry(q_machine=self.qm, wires=1, params=x[:, [0]])
                self.xlayer(q_machine=self.qm)
                self.rx_layer1(params=self.rot_param, wire=1, q_machine=self.qm)
                self.ry_layer2(q_machine=self.qm)
                self.cnot(q_machine=self.qm)
                rlt = self.measure(q_machine=self.qm)

                return rlt


        input_x = tensor.arange(1, bsz * 2 + 1,
                                dtype=pyvqnet.kfloat32).reshape([bsz, 2])
        input_x.requires_grad = True

        qunatum_model = QModel(num_wires=2, dtype=pyvqnet.kcomplex64)

        batch_y = qunatum_model(input_x)
        batch_y.backward()

        print(batch_y)

        # [[5.3798223],
        #  [7.129416 ],
        #  [0.70283  ]]
        # <QTensor [3, 1] DEV_CPU float32>



ShotConfig
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.ShotConfig(shots=None, seed=None, readout=None)

    有限采样测量的配置:采样次数、采样随机种子和可选的读出错误。``ShotConfig`` 是不可修改的数据类(dataclass),供 ``sample_counts`` 和 ``sample_expectation`` 使用。

    :param shots: 采样次数,正整数;默认值:None,表示不做采样,``sample_expectation`` 直接返回解析计算的期望。
    :param seed: 采样随机种子,整数;默认值:None,此时使用最近一次 ``pyvqnet.utils.set_random_seed`` 设置的种子,未设置过时使用随机熵。
    :param readout: 读出错误,默认值:None,表示没有读出错误。可以是一个 ``[0, 1]`` 内的浮点数 :math:`p`,表示读出时每个量子比特以概率 :math:`p` 发生翻转;也可以是 :math:`2 \times 2` 的行随机矩阵 ``[[P(读到0 | 真值0), P(读到1 | 真值0)], [P(读到0 | 真值1), P(读到1 | 真值1)]]``,每一行的和必须为1。
    :return: 一个 ShotConfig 配置实例。

    Example::

        from pyvqnet.qnn.vqc import ShotConfig

        config = ShotConfig(shots=2000, seed=123, readout=0.02)
        print(config.shots, config.seed, config.readout)
        print(ShotConfig())                       # analytic path: no sampling
        print(ShotConfig(shots=1000, readout=[[0.98, 0.02], [0.05, 0.95]]).readout)

        # 2000 123 0.02
        # ShotConfig(shots=None, seed=None, readout=None)
        # ((0.98, 0.02), (0.05, 0.95))


sample_counts
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.sample_counts(qm, wires, config)

    对 ``qm`` 当前量子态在量子比特 ``wires`` 上做有限次采样,返回各个计算基矢结果出现的次数。

    :param qm: ``pyvqnet.qnn.vqc.QMachine`` 模拟器。
    :param wires: 采样的量子比特索引,整数或者列表;``None`` 表示模拟器的全部量子比特。
    :param config: ``ShotConfig`` 配置,其中 ``shots`` 必须是正整数,否则报 ``ValueError``。
    :return: ``numpy.ndarray``,数据类型 ``int64``,形状为 ``(b, 2**len(wires))``,每一行是对应样本的各个基矢结果的次数,行之和为 ``shots``,其中b为 ``qm.reset_states(b)`` 的批处理数量b。测量部分比特时,基矢按照 ``wires`` 列表顺序排列(第一个比特为最高位);测量全部比特时,基矢按照比特索引升序排列。

    Example::

        from pyvqnet.qnn.vqc import QMachine, Probability, ShotConfig, ry, sample_counts
        from pyvqnet import tensor

        qm = QMachine(2)
        qm.reset_states(2)
        ry(q_machine=qm, wires=0, params=tensor.QTensor([[0.9], [0.3]]))

        config = ShotConfig(shots=2000, seed=123)
        counts = sample_counts(qm, [0], config)
        print(type(counts).__name__, counts.dtype, counts.shape)
        print(counts)
        print(counts / config.shots)
        print(Probability(wires=[0])(q_machine=qm))

        # ndarray int64 (2, 2)
        # [[1607  393]
        #  [1957   43]]
        # [[0.8035 0.1965]
        #  [0.9785 0.0215]]
        #
        # [[0.810805 ,0.189195 ],
        #  [0.9776682,0.0223318]]
        # <QTensor [2, 2] DEV_CPU float32>


sample_expectation
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.sample_expectation(qm, obs, config)

    估计 Pauli 算符组合 ``obs`` 的期望。

    ``config.shots`` 为 ``None`` 或者 ``config.readout`` 没有读出错误(``None``、``0.0`` 或者单位矩阵)时,直接返回解析计算的期望;否则从带读出错误的有限次采样结果中估计,此时 ``obs`` 中只能包含 ``Z`` 和 ``I`` 算符,包含 ``X`` 或者 ``Y`` 时报 ``ValueError``。

    :param qm: ``pyvqnet.qnn.vqc.QMachine`` 模拟器。
    :param obs: 字典格式的 Pauli 算符组合,与 ``MeasureAll`` 的 ``obs`` 相同,例如 ``{"Z0": 1.0, "Z1 Z2": -0.5}``。
    :param config: ``ShotConfig`` 配置。
    :return: 期望估计值,``float``;批处理时为各个样本的平均。

    Example::

        from pyvqnet.qnn.vqc import QMachine, ShotConfig, cnot, ry, sample_expectation
        from pyvqnet import tensor

        qm = QMachine(2)
        qm.reset_states(1)
        ry(q_machine=qm, wires=0, params=tensor.QTensor([0.9]))
        cnot(q_machine=qm, wires=[0, 1])

        # no readout error: the analytic expectation is returned
        print(sample_expectation(qm, {"Z0": 1.0}, ShotConfig(shots=2000, seed=123)))

        # with a readout error the estimate comes from finite shots
        config = ShotConfig(shots=200000, seed=123, readout=0.05)
        print(sample_expectation(qm, {"Z0": 1.0}, config))
        print(sample_expectation(qm, {"Z0": 1.0, "Z1": 1.0}, config))

        # 0.6216100454330444
        # 0.5586
        # 1.12107


fit_shot_scaling
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.fit_shot_scaling(counts_by_shots)

    拟合有限采样标准误随采样次数的标度指数:对每个采样次数,由多次独立重复的计数估计第一个基矢结果出现的频率,取其标准误,返回标准误的对数关于采样次数对数的线性拟合斜率。理想情况下标准误正比于 :math:`1/\sqrt{\text{shots}}`,斜率约为 ``-0.5``。

    :param counts_by_shots: 字典,键为采样次数 ``shots``,值为 ``sample_counts`` 的结果沿首个维度堆叠的数组,形状 ``(重复次数, b, 2**len(wires))``;至少需要两个采样次数,每个采样次数至少两次重复。
    :return: 拟合得到的斜率,``float``。

    Example::

        import numpy as np
        from pyvqnet.qnn.vqc import QMachine, ShotConfig, ry, fit_shot_scaling, sample_counts
        from pyvqnet import tensor

        qm = QMachine(1)
        qm.reset_states(1)
        ry(q_machine=qm, wires=0, params=tensor.QTensor([0.9]))

        counts_by_shots = {}
        for shots in (100, 1000, 10000):
            repetitions = [
                sample_counts(qm, [0], ShotConfig(shots=shots, seed=1000 + index))
                for index in range(20)
            ]
            counts_by_shots[shots] = np.stack(repetitions)
            print(shots, counts_by_shots[shots].shape, counts_by_shots[shots].sum(axis=1)[:3])

        # the standard error scales as shots ** slope, the slope is about -0.5
        print(fit_shot_scaling(counts_by_shots))

        # 100 (20, 1, 2) [[75 25]
        #  [79 21]
        #  [80 20]]
        # 1000 (20, 1, 2) [[823 177]
        #  [825 175]
        #  [809 191]]
        # 10000 (20, 1, 2) [[8145 1855]
        #  [8088 1912]
        #  [8075 1925]]
        # -0.5033285530545737

常用量子变分线路模板
=======================================

VQC_HardwareEfficientAnsatz
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.VQC_HardwareEfficientAnsatz(n_qubits,single_rot_gate_list,entangle_gate="CNOT",entangle_rules='linear',depth=1,initial=None,dtype=None)

    一个含可训练参数的变分量子线路类,实现了论文介绍的Hardware Efficient Ansatz的实现: `Hardware-efficient Variational Quantum Eigensolver for Small Molecules <https://arxiv.org/pdf/1704.05018.pdf>`__ 。

    :param n_qubits: 量子比特数。
    :param single_rot_gate_list: 单个量子比特旋转门列表由一个或多个作用于每个量子比特的旋转门构成。目前支持 Rx、Ry、Rz。
    :param entangle_gate: 非参数化纠缠门。支持 CNOT、CZ。默认值: CNOT。
    :param entangle_rules: 纠缠门在电路中的使用方式。"linear" 表示纠缠门将作用于每个相邻的量子比特。'all' 表示纠缠门将作用于任意两个量子比特。默认值: "linear"。
    :param depth: 假设的深度, 默认值: 1。
    :param initial: 使用initial 初始化所有其中参数逻辑门的参数, 默认值: None, 此模块将随机初始化参数。
    :param dtype: 参数的数据类型,默认值: None,使用 float32。
    :return: 一个 VQC_HardwareEfficientAnsatz 实例。



    Example::

        from pyvqnet.nn import Module,Linear,ModuleList
        from pyvqnet.qnn.vqc import VQC_HardwareEfficientAnsatz,RZZ,RZ
        from pyvqnet.qnn.vqc import Probability,QMachine
        from pyvqnet import tensor

        class QM(Module):
            def __init__(self, name=""):
                super().__init__(name)
                self.linearx = Linear(4,2)
                self.ansatz = VQC_HardwareEfficientAnsatz(4, ["rx", "RY", "rz"],
                                            entangle_gate="cnot",
                                            entangle_rules="linear",
                                            depth=2)
                self.encode1 = RZ(wires=0)
                self.encode2 = RZ(wires=1)
                self.measure = Probability(wires=[0,2])
                self.device = QMachine(4)
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(x.shape[0])
                y = self.linearx(x)
                self.encode1(params = y[:, [0]],q_machine = self.device,)
                self.encode2(params = y[:, [1]],q_machine = self.device,)
                self.ansatz(q_machine =self.device)
                return self.measure(q_machine =self.device)

        bz =3
        inputx = tensor.arange(1.0,bz*4+1).reshape([bz,4])
        inputx.requires_grad= True
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)

        # [[0.3075959 0.2315064 0.2491432 0.2117545]
        #  [0.3075958 0.2315062 0.2491433 0.2117546]
        #  [0.3075958 0.2315062 0.2491432 0.2117545]]

VQC_BasicEntanglerTemplate
-----------------------------------------

.. py:class:: pyvqnet.qnn.vqc.VQC_BasicEntanglerTemplate(num_layers=1, num_qubits=1, rotation="RX", initial=None, dtype=None)

    由每个量子位上的单参数单量子位旋转组成的层,后跟一个闭合链或环组合的多个CNOT 门构成的类。

    CNOT 门环将每个量子位与其邻居连接起来,最后一个量子位被认为是第一个量子位的邻居。

    :param num_layers: 量子比特线路层数。
    :param num_qubits: 量子比特数,默认为1。
    :param rotation: 使用单参数单量子比特门,``RX`` 被用作默认值。
    :param initial: 使用initial 初始化所有其中参数逻辑门的参数, 默认值: None, 此模块将随机初始化参数。
    :param dtype: 参数的数据类型,默认值: None,使用 float32。
    :return: 返回一个含可训练参数的VQC_BasicEntanglerTemplate实例。

    Example::

        from pyvqnet.nn import Module, Linear, ModuleList
        from pyvqnet.qnn.vqc import VQC_BasicEntanglerTemplate, RZZ, RZ
        from pyvqnet.qnn.vqc import Probability, QMachine
        from pyvqnet import tensor


        class QM(Module):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_BasicEntanglerTemplate(2,
                                                    4,
                                                    "rz",
                                                    initial=tensor.ones([1, 1]))

                self.measure = Probability(wires=[0, 2])
                self.device = QMachine(4)

            def forward(self,x, *args, **kwargs):

                self.ansatz(q_machine=self.device)
                return self.measure(q_machine=self.device)

        bz = 1
        inputx = tensor.arange(1.0, bz * 4 + 1).reshape([bz, 4])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)

        # [[1.0000002 0.        0.        0.       ]]


VQC_StronglyEntanglingTemplate
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.VQC_StronglyEntanglingTemplate(num_layers=1, num_qubits=1, ranges=None, initial=None, dtype=None)

    由单个量子比特旋转和纠缠器组成的层,参考 `circuit-centric classifier design <https://arxiv.org/abs/1804.00633>`__ .

 
    :param num_layers: 重复层数,默认值: 1。
    :param num_qubits: 量子比特数,默认值: 1。
    :param ranges: 决定每个后续层的范围超参数的序列。默认值: None,对第 :math:`l` 层和 :math:`M` 个量子比特使用 :math:`r=l \mod M`。
    :param initial: 使用initial 初始化所有其中参数逻辑门的参数,默认值: None,此模块将随机初始化参数。
    :param dtype: 参数的数据类型,默认值: None,使用 float32。
    :return: VQC_StronglyEntanglingTemplate 实例


    Example::

        from pyvqnet.nn import Module
        from pyvqnet.qnn.vqc import VQC_StronglyEntanglingTemplate
        from pyvqnet.qnn.vqc import Probability, QMachine
        from pyvqnet import tensor


        class QM(Module):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_StronglyEntanglingTemplate(2,
                                                    4,
                                                    None,
                                                    initial=tensor.ones([1, 1]))

                self.measure = Probability(wires=[0, 1])
                self.device = QMachine(4)

            def forward(self,x, *args, **kwargs):

                self.ansatz(q_machine=self.device)
                return self.measure(q_machine=self.device)

        bz = 1
        inputx = tensor.arange(1.0, bz * 4 + 1).reshape([bz, 4])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)

        # [[0.3745951 0.154298  0.059156  0.4119509]]


VQC_QuantumEmbedding
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.VQC_QuantumEmbedding( num_repetitions_input, depth_input, num_unitary_layers, num_repetitions,initial = None,dtype = None,name= "")

    使用 RZ,RY,RZ 创建变分量子电路,将经典数据编码为量子态。
    参考 `Quantum embeddings for machine learning <https://arxiv.org/abs/2001.03622>`_。

 
    :param num_repetitions_input: 子模块中输入编码的重复次数。
    :param depth_input: 输入维数。
    :param num_unitary_layers: 变分量子门的重复次数。
    :param num_repetitions: 子模块的重复次数。
    :param initial: 参数初始化值,默认为None
    :param dtype: 参数的类型,默认 None,使用float32.
    :param name: 类的名字

    Example::

        from pyvqnet.nn import Module
        from pyvqnet.qnn.vqc import VQC_QuantumEmbedding
        from pyvqnet.qnn.vqc import  QMachine,MeasureAll
        from pyvqnet import tensor
        import pyvqnet
        depth_input = 2
        num_repetitions = 2
        num_repetitions_input = 2
        num_unitary_layers = 2
        nq = depth_input * num_repetitions_input
        bz = 12

        class QM(Module):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_QuantumEmbedding(num_repetitions_input, depth_input,
                                                num_unitary_layers,
                                                num_repetitions, initial=tensor.full([1],12.0),dtype=pyvqnet.kfloat64)

                self.measure = MeasureAll(obs = {f"Z{nq-1}":1})
                self.device = QMachine(nq,dtype=pyvqnet.kcomplex128)

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(x.shape[0])
                self.ansatz(x,q_machine=self.device)
                return self.measure(q_machine=self.device)

        inputx = tensor.arange(1.0, bz * depth_input + 1,
                                dtype=pyvqnet.kfloat64).reshape([bz, depth_input])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)

        # [[-0.2539548]
        #  [-0.1604787]
        #  [ 0.1492931]
        #  [-0.1711956]
        #  [-0.1577133]
        #  [ 0.1396999]
        #  [ 0.016864 ]
        #  [-0.0893069]
        #  [ 0.1897014]
        #  [ 0.0941301]
        #  [ 0.0550722]
        #  [ 0.2408579]]

ExpressiveEntanglingAnsatz
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.ExpressiveEntanglingAnsatz(type: int, num_wires: int, depth: int, dtype=None, name: str = "")

    论文 `Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms <https://arxiv.org/pdf/1905.10876.pdf>`_ 中的 19 种不同的ansatz。

    :param type: 电路类型从 1 到 19,共19种线路。
    :param num_wires: 量子比特数。
    :param depth: 电路深度。
    :param dtype: 参数的数据类型, 默认值: None, 使用 float32。
    :param name: 名字,默认"".

    :return:
        一个 ExpressiveEntanglingAnsatz 实例

    Example::

        from pyvqnet.qnn.vqc  import *
        import pyvqnet
        pyvqnet.utils.set_random_seed(42)
        from pyvqnet.nn import Module
        class QModel(Module):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype,save_ir=True)
                self.c1 = ExpressiveEntanglingAnsatz(13,3,2)
                self.measure = MeasureAll(obs = {
                    "Z1":1
                })

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])
                self.c1(q_machine = self.qm)
                rlt = self.measure(q_machine=self.qm)
                return rlt
            

        input_x = tensor.QTensor([[0.1, 0.2, 0.3]])

        input_x.requires_grad = True

        qunatum_model = QModel(num_wires=3, dtype=pyvqnet.kcomplex64)

        batch_y = qunatum_model(input_x)
        z = vqc_to_originir_list(qunatum_model)
        for zi in z:
            print(zi)
        batch_y.backward()
        print(batch_y)



VQC_BasisEmbedding
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_BasisEmbedding(basis_state,q_machine,wires=None)

    将二进制特征 ``basis_state`` 编码为 ``q_machine`` 的基态。

    例如, 对于 ``basis_state=([0, 1, 1])``, 在量子系统下其基态为 :math:`|011 \rangle`。

    :param basis_state:  ``(n)`` 大小的二进制输入。
    :param q_machine: 量子虚拟机设备。
    :param wires: 编码作用的量子比特,默认为 None,使用 ``q_machine`` 的全部量子比特。

    Example::
        
        from pyvqnet.qnn.vqc import VQC_BasisEmbedding,QMachine
        qm  = QMachine(3)
        VQC_BasisEmbedding(basis_state=[1,1,0],q_machine=qm)
        print(qm.states)

        # [[[[0.+0.j 0.+0.j]
        #    [0.+0.j 0.+0.j]]
        # 
        #   [[0.+0.j 0.+0.j]
        #    [1.+0.j 0.+0.j]]]]


VQC_AngleEmbedding
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_AngleEmbedding(input_feat, wires, q_machine: pyvqnet.qnn.vqc.QMachine, rotation: str = "X")

    将 :math:`N` 特征编码到 ``q_machine`` :math:`n` 量子比特的旋转角度中, 其中 :math:`N \leq n`。

    旋转可以选择为 : 'X' , 'Y' , 'Z', 如 ``rotation`` 的参数定义为:

    * ``rotation='X'`` 将特征用作RX旋转的角度。

    * ``rotation='Y'`` 将特征用作RY旋转的角度。

    * ``rotation='Z'`` 将特征用作RZ旋转的角度。

     ``wires`` 代表旋转门在量子比特上的idx。

    :param input_feat: 表示参数的数组。
    :param wires: 量子比特idx。
    :param q_machine: 量子虚拟机设备。
    :param rotation: 旋转门,默认为“X”。
    

    Example::

        from pyvqnet.qnn.vqc import VQC_AngleEmbedding, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(2)
        VQC_AngleEmbedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='X')
        
        print(qm.states)
        # [[[ 0.398068 +0.j         0.       -0.2174655j]
        #   [ 0.       -0.7821081j -0.4272676+0.j       ]]]

        VQC_AngleEmbedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='Y')

        print(qm.states)
        # [[[-0.0240995+0.6589843j  0.4207355+0.2476033j]
        #   [ 0.4042482-0.2184162j  0.       -0.3401631j]]]

        VQC_AngleEmbedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='Z')

        print(qm.states)

        # [[[0.659407 +0.0048471j 0.4870554-0.0332093j]
        #   [0.4569675+0.047989j  0.340018 +0.0099326j]]]

VQC_AmplitudeEmbedding
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_AmplitudeEmbedding(input_feature, q_machine)

    将 :math:`2^n` 特征编码为 ``q_machine``  :math:`n` 量子比特的振幅向量。

    :param input_feature: 表示参数的numpy数组。
    :param q_machine: 量子虚拟机设备。
    

    Example::

        from pyvqnet.qnn.vqc import VQC_AmplitudeEmbedding, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(3)
        VQC_AmplitudeEmbedding(QTensor([3.2,-2,-2,0.3,12,0.1,2,-1]), q_machine=qm)
        print(qm.states)

        # [[[[ 0.2473717+0.j -0.1546073+0.j]
        #    [-0.1546073+0.j  0.0231911+0.j]]
        # 
        #   [[ 0.9276441+0.j  0.0077304+0.j]
        #    [ 0.1546073+0.j -0.0773037+0.j]]]]

VQC_IQPEmbedding
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_IQPEmbedding(input_feat, q_machine: pyvqnet.qnn.vqc.QMachine, rep: int = 1)

    使用IQP线路的对角门将 :math:`n` 特征编码为 ``q_machine``  :math:`n` 量子比特。

    编码是由 `Havlicek et al. (2018) <https://arxiv.org/pdf/1804.11326.pdf>`_ 提出。

    通过指定 ``rep`` ,可以重复基本IQP线路。

    :param input_feat: 表示参数的数组。
    :param q_machine: 量子虚拟机设备。
    :param rep: 重复量子线路块次数,默认次数为1。
    

    Example::

        from pyvqnet.qnn.vqc import VQC_IQPEmbedding, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(3)
        VQC_IQPEmbedding(QTensor([3.2,-2,-2]), q_machine=qm)
        print(qm.states)        
        
        # [[[[ 0.0309356-0.3521973j  0.3256442+0.1376801j]
        #    [ 0.3256442+0.1376801j  0.2983474+0.1897071j]]
        # 
        #   [[ 0.0309356+0.3521973j -0.3170519-0.1564546j]
        #    [-0.3170519-0.1564546j -0.2310978-0.2675701j]]]]


VQC_RotCircuit
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_RotCircuit(q_machine, wire, params)

    在 ``q_machine`` 的比特 ``wire`` 上使用 ``params`` 进行单量子比特旋转操作。

    .. math::

        R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
        e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
        e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.


    :param q_machine: 量子虚拟机设备。
    :param wire: 量子比特索引。
    :param params: 表示参数  :math:`[\phi, \theta, \omega]`。
    

    Example::

        from pyvqnet.qnn.vqc import VQC_RotCircuit, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(3)
        VQC_RotCircuit(q_machine=qm, wire=[1],params=QTensor([2.0,1.5,2.1]))
        print(qm.states)

        # [[[[-0.3373617-0.6492732j  0.       +0.j       ]
        #    [ 0.6807868-0.0340677j  0.       +0.j       ]]
        # 
        #   [[ 0.       +0.j         0.       +0.j       ]
        #    [ 0.       +0.j         0.       +0.j       ]]]]

VQC_CRotCircuit
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_CRotCircuit(para,control_wire,rot_wire,q_machine)

    在 ``q_machine`` 的比特 ``rot_wire`` 以及受控比特 ``control_wire`` 上使用 ``params`` 进行受控Rot操作符。

    .. math:: CR(\phi, \theta, \omega) = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0\\
            0 & 0 & e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2)\\
            0 & 0 & e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.
    
    :param para: 表示参数的数组。
    :param control_wire: 控制量子比特索引。
    :param rot_wire: Rot量子比特索引。
    :param q_machine: 量子虚拟机设备。
    

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc import VQC_CRotCircuit
        from pyvqnet.qnn.vqc import QMachine, MeasureAll
        p = QTensor([2, 3, 4.0])
        qm = QMachine(2)
        VQC_CRotCircuit(p, 0, 1, qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)

        # [[0.9999999]]




VQC_Controlled_Hadamard
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_Controlled_Hadamard(wires, q_machine)

    在 ``q_machine`` 的比特 ``wire`` 上使用受控Hadamard逻辑门

    .. math:: CH = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
            0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
        \end{bmatrix}.

    :param wires: 量子比特索引列表, 第一位是控制比特, 列表长度为2。
    :param q_machine: 量子虚拟机设备。
    

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc import VQC_Controlled_Hadamard
        from pyvqnet.qnn.vqc import QMachine, MeasureAll
        p = QTensor([0.2, 3, 4.0])

        qm = QMachine(3)

        VQC_Controlled_Hadamard([1, 0], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)

        # [[1.]]

VQC_CCZ
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_CCZ(wires, q_machine)

    在 ``q_machine`` 的比特 ``wire`` 上使用受控-受控-Z (controlled-controlled-Z) 逻辑门。

    .. math::

        CCZ =
        \begin{pmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & -1
        \end{pmatrix}
    
    :param wires: 量子比特下标列表,第一位是控制比特。列表长度为3。
    :param q_machine: 量子虚拟机设备。
    

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc import VQC_CCZ
        from pyvqnet.qnn.vqc import QMachine, MeasureAll
        p = QTensor([0.2, 3, 4.0])

        qm = QMachine(3)

        VQC_CCZ([1, 0, 2], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)

        # [[0.9999999]]


VQC_FermionicSingleExcitation
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_FermionicSingleExcitation(weight, wires, q_machine)

    在 ``q_machine`` 的比特 ``wire`` 上使用由 ``weight`` 构成的泡利矩阵的张量积求幂的耦合簇单激励算子。矩阵形式下式给出:

    .. math::

        \hat{U}_{pr}(\theta) = \mathrm{exp} \{ \theta_{pr} (\hat{c}_p^\dagger \hat{c}_r
        -\mathrm{H.c.}) \},

    :param weight:  量子比特p上的参数, 只有一个元素.
    :param wires: 表示区间[r, p]中的量子比特索引子集。最小长度必须为2。第一索引值被解释为r,最后一个索引值被解释为p。
                中间的索引被CNOT门作用,以计算量子位集的奇偶校验。
    :param q_machine: 量子虚拟机设备。

    

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc import VQC_FermionicSingleExcitation
        from pyvqnet.qnn.vqc import QMachine, MeasureAll
        qm = QMachine(3)
        p0 = QTensor([0.5])

        VQC_FermionicSingleExcitation(p0, [1, 0, 2], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)

        # [[0.9999998]]


VQC_FermionicDoubleExcitation
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_FermionicDoubleExcitation(weight, wires1, wires2, q_machine)

    在 ``q_machine`` 的比特 ``wire1`` , ``wire2`` 上使用由 ``weight`` 对泡利矩阵的张量积求幂的耦合聚类双激励算子,矩阵形式由下式给出:

    .. math::

        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \{ \theta (\hat{c}_p^\dagger \hat{c}_q^\dagger
        \hat{c}_r \hat{c}_s - \mathrm{H.c.}) \},

    其中 :math:`\hat{c}` 和 :math:`\hat{c}^\dagger` 是费米子湮灭和
    创建运算符和索引 :math:`r, s` 和 :math:`p, q` 在占用的和
    分别为空分子轨道。 使用 `Jordan-Wigner 变换
    <https://arxiv.org/abs/1208.5986>`_ 上面定义的费米子算子可以写成
    根据 Pauli 矩阵(有关更多详细信息,请参见
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_)

    .. math::

        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \Big\{
        \frac{i\theta}{8} \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1}
        \hat{Z}_a (\hat{X}_s \hat{X}_r \hat{Y}_q \hat{X}_p +
        \hat{Y}_s \hat{X}_r \hat{Y}_q \hat{Y}_p +\\ \hat{X}_s \hat{Y}_r \hat{Y}_q \hat{Y}_p +
        \hat{X}_s \hat{X}_r \hat{X}_q \hat{Y}_p - \mathrm{H.c.}  ) \Big\}

    :param weight: 可变参数
    :param wires1: 代表的量子比特的索引列表区间 [s, r] 中占据量子比特的子集。第一个索引被解释为 s,最后一索引被解释为 r。 CNOT 门对中间的索引进行操作,以计算一组量子位的奇偶性。
    :param wires2: 代表的量子比特的索引列表区间 [q, p] 中占据量子比特的子集。第一根索引被解释为 q,最后一索引被解释为 p。 CNOT 门对中间的索引进行操作,以计算一组量子位的奇偶性。
    :param q_machine: 量子虚拟机设备。

    

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc import VQC_FermionicDoubleExcitation
        from pyvqnet.qnn.vqc import QMachine, MeasureAll
        qm = QMachine(5)
        p0 = QTensor([0.5])

        VQC_FermionicDoubleExcitation(p0, [0, 1], [2, 3], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)
        
        # [[0.9999998]]

VQC_UCCSD
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_UCCSD(weights, wires, s_wires, d_wires, init_state, q_machine)

    在 ``q_machine`` 上实现酉耦合簇单激发和双激发拟设(UCCSD)。UCCSD 是 VQE 拟设,通常用于运行量子化学模拟。

    在一阶 Trotter 近似内,UCCSD 酉函数由下式给出:

    .. math::

        \hat{U}(\vec{\theta}) =
        \prod_{p > r} \mathrm{exp} \Big\{\theta_{pr}
        (\hat{c}_p^\dagger \hat{c}_r-\mathrm{H.c.}) \Big\}
        \prod_{p > q > r > s} \mathrm{exp} \Big\{\theta_{pqrs}
        (\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s-\mathrm{H.c.}) \Big\}

    其中 :math:`\hat{c}` 和 :math:`\hat{c}^\dagger` 是费米子湮灭和
    创建运算符和索引 :math:`r, s` 和 :math:`p, q` 在占用的和
    分别为空分子轨道。(更多细节见
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_):


    :param weights: 包含参数的大小 ``(len(s_wires)+ len(d_wires))`` 张量
        :math:`\theta_{pr}` 和 :math:`\theta_{pqrs}` 输入 Z 旋转
        ``FermionicSingleExcitation`` 和 ``FermionicDoubleExcitation`` 。
    :param wires: 模板作用的量子比特索引
    :param s_wires: 包含量子比特索引的列表序列 ``[r,...,p]``
        由单一激发产生
        :math:`\vert r, p \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF} \rangle`,
        其中 :math:`\vert \mathrm{HF} \rangle` 表示 Hartee-Fock 参考态。
    :param d_wires: 列表序列,每个列表包含两个列表
        指定索引 ``[s, ...,r]`` 和 ``[q,..., p]`` 
        定义双激励 :math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r\hat{c}_s \vert \mathrm{HF} \rangle` 。
    :param init_state: 长度 ``len(wires)`` occupation-number vector 表示
        高频状态。 ``init_state`` 在量子比特初始化状态。
    :param q_machine: 量子虚拟机设备。
    
    
    Example::

        from pyvqnet.qnn.vqc import VQC_UCCSD, QMachine, MeasureAll
        from pyvqnet.tensor import QTensor
        p0 = QTensor([2, 0.5, -0.2, 0.3, -2, 1, 3, 0])
        s_wires = [[0, 1, 2], [0, 1, 2, 3, 4], [1, 2, 3], [1, 2, 3, 4, 5]]
        d_wires = [[[0, 1], [2, 3]], [[0, 1], [2, 3, 4, 5]], [[0, 1], [3, 4]],
                [[0, 1], [4, 5]]]
        qm = QMachine(6)

        VQC_UCCSD(p0, range(6), s_wires, d_wires, QTensor([1.0, 1, 0, 0, 0, 0]), qm)
        m = MeasureAll(obs={"Z1": 1})
        exp = m(q_machine=qm)
        print(exp)

        # [[0.963802]]

VQC_ZFeatureMap
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_ZFeatureMap(input_feat, q_machine: pyvqnet.qnn.vqc.QMachine, data_map_func=None, rep: int = 2)

    在 ``q_machine`` 上运行一阶泡利 Z 演化电路。

    对于 3 个量子位和 2 次重复,电路表示为:

    .. parsed-literal::

        ┌───┐┌──────────────┐┌───┐┌──────────────┐
        ┤ H ├┤ U1(2.0*x[0]) ├┤ H ├┤ U1(2.0*x[0]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[1]) ├┤ H ├┤ U1(2.0*x[1]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[2]) ├┤ H ├┤ U1(2.0*x[2]) ├
        └───┘└──────────────┘└───┘└──────────────┘
    
    泡利弦固定为 ``Z``。 因此,一阶展开将是一个没有纠缠门的电路。

    :param input_feat: 表示输入参数的数组。
    :param q_machine: 量子虚拟机。
    :param data_map_func: 参数映射矩阵, 为可调用函数, 设计方式为: ``data_map_func = lambda x: x``。
    :param rep: 模块重复次数。
    
    Example::

        from pyvqnet.qnn.vqc import VQC_ZFeatureMap, QMachine,hadamard
        from pyvqnet.tensor import QTensor
        qm = QMachine(3)
        for i in range(3):
            hadamard(q_machine=qm, wires=[i])
        VQC_ZFeatureMap(input_feat=QTensor([[0.1,0.2,0.3]]),q_machine = qm)
        print(qm.states)
        
        # [[[[0.3535534+0.j        0.2918002+0.1996312j]
        #    [0.3256442+0.1376801j 0.1910257+0.2975049j]]
        # 
        #   [[0.3465058+0.0702402j 0.246323 +0.2536236j]
        #    [0.2918002+0.1996312j 0.1281128+0.3295255j]]]]

VQC_ZZFeatureMap
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_ZZFeatureMap(input_feat, q_machine: pyvqnet.qnn.vqc.QMachine, data_map_func=None, entanglement: Union[str, List[List[int]],Callable[[int], List[int]]] = "full",rep: int = 2)

    在 ``q_machine`` 上运行二阶 Pauli-Z 演化电路。

    对于 3 个量子位、1 个重复和线性纠缠,电路表示为:

    .. parsed-literal::

        ┌───┐┌─────────────────┐
        ┤ H ├┤ U1(2.0*φ(x[0])) ├──■────────────────────────────■────────────────────────────────────
        ├───┤├─────────────────┤┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[1])) ├┤ X ├┤ U1(2.0*φ(x[0],x[1])) ├┤ X ├──■────────────────────────────■──
        ├───┤├─────────────────┤└───┘└──────────────────────┘└───┘┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[2])) ├──────────────────────────────────┤ X ├┤ U1(2.0*φ(x[1],x[2])) ├┤ X ├
        └───┘└─────────────────┘                                  └───┘└──────────────────────┘└───┘
    
    其中 ``φ`` 是经典的非线性函数,如果输入两个值则 ``φ(x,y) = (pi - x)(pi - y)``, 输入一个则为 ``φ(x) = x``, 用 ``data_map_func`` 表示如下:
    
    .. code-block::
        
        def data_map_func(x):
            coeff = x if x.shape[-1] == 1 else ft.reduce(lambda x, y: (np.pi - x) * (np.pi - y), x)
            return coeff

    :param input_feat: 表示输入参数的数组。
    :param q_machine: 量子虚拟机。
    :param data_map_func: 参数映射矩阵, 为可调用函数。 
    :param entanglement: 指定的纠缠结构。
    :param rep: 模块重复次数。
    
    Example::

        from pyvqnet.qnn.vqc import VQC_ZZFeatureMap, QMachine
        from pyvqnet.tensor import QTensor
        qm = QMachine(3)
        VQC_ZZFeatureMap(q_machine=qm, input_feat=QTensor([[0.1,0.2,0.3]]))
        print(qm.states)

        # [[[[-0.4234843-0.0480578j -0.144067 +0.1220178j]
        #    [-0.0800646+0.0484439j -0.5512857-0.2947832j]]
        # 
        #   [[ 0.0084012-0.0050071j -0.2593993-0.2717131j]
        #    [-0.1961917-0.3470543j  0.2786197+0.0732045j]]]]

VQC_AllSinglesDoubles
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_AllSinglesDoubles(weights, q_machine: pyvqnet.qnn.vqc.QMachine, hf_state, wires, singles=None, doubles=None)

    在这种情况下,我们有四个单激发和双激发来保留 Hartree-Fock 态的总自旋投影。 

    由此产生的酉矩阵保留了粒子数量,并在初始 Hartree-Fock 状态和编码多激发配置的其他状态的叠加中准备了n量子位系统。
      
    :param weights: 大小为 ``(len(singles) + len(doubles),)`` 的QTensor,包含按顺序进入 vqc.qCircuit.single_excitation 和 vqc.qCircuit.double_excitation 操作的角度
    :param q_machine: 量子虚拟机。
    :param hf_state: 代表 Hartree-Fock 状态的长度 ``len(wires)`` 占用数向量, ``hf_state`` 用于初始化线路。
    :param wires: 作用的量子位。
    :param singles: 具有single_exitation操作所作用的两个量子位索引的列表序列。
    :param doubles: 具有double_exitation操作所作用的两个量子位索引的列表序列。

    例如,两个电子和六个量子位情况下的量子电路如下图所示:
    
.. image:: ./images/all_singles_doubles.png
    :width: 600 px
    :align: center

|

    Example::

        from pyvqnet.qnn.vqc import VQC_AllSinglesDoubles, QMachine
        from pyvqnet.tensor import QTensor
        qubits = 4
        qm = QMachine(qubits)

        VQC_AllSinglesDoubles(q_machine=qm, weights=QTensor([0.55, 0.11, 0.53]), 
                              hf_state = QTensor([1,1,0,0]), singles=[[0, 2], [1, 3]], doubles=[[0, 1, 2, 3]], wires=[0,1,2,3])
        print(qm.states)
        
        # [ 0.        +0.j  0.        +0.j  0.        +0.j -0.23728043+0.j
        #   0.        +0.j  0.        +0.j -0.27552837+0.j  0.        +0.j
        #   0.        +0.j -0.12207296+0.j  0.        +0.j  0.        +0.j
        #   0.9235152 +0.j  0.        +0.j  0.        +0.j  0.        +0.j]


VQC_BasisRotation
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_BasisRotation(q_machine: pyvqnet.qnn.vqc.QMachine, wires, unitary_matrix: QTensor, check=False)

    在 ``q_machine`` 上实现一个电路,执行精确的单体基础旋转。

    :class:`~.vqc.VQC_BasisRotation` 执行以下由 `arXiv:1711.04789 <https://arxiv.org/abs/1711.04789>`_\ 中给出的单粒子费米子确定的酉变换 :math:`U(u)`
    
    .. math::

        U(u) = \exp{\left( \sum_{pq} \left[\log u \right]_{pq} (a_p^\dagger a_q - a_q^\dagger a_p) \right)}.
    
    :math:`U(u)` 通过使用论文 `Optica, 3, 1460 (2016) <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_\ 中给出的方案。
    

    :param q_machine: 量子虚拟机。
    :param wires: 作用的量子位。
    :param unitary_matrix: 指定基础变换的矩阵。
    :param check: 检测 `unitary_matrix` 是否为酉矩阵。

    Example::

        from pyvqnet.qnn.vqc import VQC_BasisRotation, QMachine, hadamard, isingzz
        from pyvqnet.tensor import QTensor
        import numpy as np
        V = np.array([[0.73678+0.27511j, -0.5095 +0.10704j, -0.06847+0.32515j],
                      [0.73678+0.27511j, -0.5095 +0.10704j, -0.06847+0.32515j],
                      [-0.21271+0.34938j, -0.38853+0.36497j,  0.61467-0.41317j]])

        eigen_vals, eigen_vecs = np.linalg.eigh(V)
        umat = eigen_vecs.T
        wires = range(len(umat))
        
        qm = QMachine(len(umat))

        for i in range(len(umat)):
            hadamard(q_machine=qm, wires=i)
        isingzz(q_machine=qm, params=QTensor([0.55]), wires=[0,2])
        VQC_BasisRotation(q_machine=qm, wires=wires,unitary_matrix=QTensor(umat,dtype=qm.states.dtype))
        
        print(qm.states)
        
        # [[[[ 0.3402686-0.0960063j  0.4140436-0.3069579j]
        #    [ 0.1206574+0.1982292j  0.5662895-0.0949503j]]
        # 
        #   [[-0.1715559-0.1614315j  0.1624039-0.0598041j]
        #    [ 0.0608986-0.1078906j -0.305845 +0.1773662j]]]]

vqc_qft_add_to_register
-------------------------------------

.. py:function:: pyvqnet.qnn.vqc.vqc_qft_add_to_register(q_machine, m, k)

    在 ``q_machine`` 进行如下操作:将无符号整数 `m` 编码到量子比特中,然后将 `k` 加到此量子比特上。

    .. math:: \text{Sum(k)}\vert m \rangle = \vert m + k \rangle.

    实现此幺正运算的过程如下:
    (1). 通过 将 QFT 应用于 :math:`\vert m \rangle` 状态,将状态从计算基础转换为傅里叶基础。
    (2). 使用 :math:`R_Z` 门将 :math:`j` 个量子比特旋转角度 :math:`\frac{2k\pi}{2^{j}}`,从而得到新相 :math:`\frac{2(m + k)\pi}{2^{j}}`。
    (3). 应用 QFT 逆返回计算基础并得到 :math:`m+k`。

    :param q_machine: 用于模拟的量子机。
    :param m: 嵌入寄存器的经典整数。
    :param k: 添加到寄存器的经典整数。

    :retrun: 返回目标和的二进制表示。

    .. note::

        请注意 ``q_machine`` 使用的比特数量需要足够使用X基态编码结果和的二进制值。

    Example::

        import numpy as np
        from pyvqnet.qnn.vqc import QMachine,Samples, vqc_qft_add_to_register
        dev = QMachine(4)
        vqc_qft_add_to_register(dev,3, 7)
        ma = Samples()
        y = ma(q_machine=dev)
        print(y)
        #[[1,0,1,0]]


vqc_qft_add_two_register
-------------------------------------

.. py:function:: pyvqnet.qnn.vqc.vqc_qft_add_two_register(q_machine, m, k, wires_m, wires_k, wires_solution)

    在 ``q_machine`` 进行如下操作: 将两个量子比特wires_m, wires_k中编码的无符号整数m, k进行加法,结果存在wires_solution比特上。

    .. math:: \text{Sum}_2\vert m \rangle \vert k \rangle \vert 0 \rangle = \vert m \rangle \vert k \rangle \vert m+k \rangle

    在这种情况下,我们可以将第三个寄存器(最初位于 :math:`0`)理解为一个计数器,它将计算出 :math:`m` 和 :math:`k` 加起来的单位数。二进制分解将使这变得简单。如果我们有 :math:`\vert m \rangle = \vert \overline{q_0q_1q_2} \rangle`,则如果 :math:`q_2 = 1`,则我们必须将 :math:`1` 添加到计数器,否则不添加任何内容。一般来说,如果 :math:`i`-th 量子位处于 :math:`\vert 1 \rangle` 状态,则我们应该添加 :math:`2^{n-i-1}` 个单位,否则添加 0。

    :param q_machine: 用于模拟的量子机。
    :param m: 嵌入寄存器中的经典整数作为 lhs。
    :param k: 嵌入到寄存器中的经典整数作为 rhs。
    :param wires_m: 要编码 m 的量子比特的索引。
    :param wires_k: 要编码 k 的量子比特的索引。
    :param wires_solution: 要编码解决方案的量子比特的索引。

    :retrun: 返回目标和的二进制表示。


    .. note::

        ``wires_m`` 使用的比特数量需要足够使用X基态编码 `m` 的二进制值。
        ``wires_k`` 使用的比特数量需要足够使用X基态编码 `k` 的二进制值。
        ``wires_solution`` 使用的比特数量需要足够使用X基态编码结果的二进制值。

    Example::

        import numpy as np
        from pyvqnet.qnn.vqc import QMachine,Samples, vqc_qft_add_two_register

        wires_m = [0, 1, 2]             # qubits needed to encode m
        wires_k = [3, 4, 5]             # qubits needed to encode k
        wires_solution = [6, 7, 8, 9]   # qubits needed to encode the solution
        dev = QMachine(len(wires_m) + len(wires_k) + len(wires_solution))

        vqc_qft_add_two_register(dev,3, 7, wires_m, wires_k, wires_solution)

        ma = Samples(wires=wires_solution)
        y = ma(q_machine=dev)
        print(y)


vqc_qft_mul
-------------------------------------

.. py:function:: pyvqnet.qnn.vqc.vqc_qft_mul(q_machine, m, k, wires_m, wires_k, wires_solution)

    在 ``q_machine`` 进行如下操作: 将两个量子比特wires_m, wires_k中编码的数值m, k, 进行乘法,结果存在wires_solution比特上。

    .. math:: \text{Mul}\vert m \rangle \vert k \rangle \vert 0 \rangle = \vert m \rangle \vert k \rangle \vert m\cdot k \rangle

    :param q_machine: 用于模拟的量子机。
    :param m: 嵌入寄存器中的经典整数,作为左侧。
    :param k: 嵌入寄存器中的经典整数,作为右侧。
    :param wires_m: 要编码 m 的量子比特索引。
    :param wires_k: 要编码 k 的量子比特索引。
    :param wires_solution: 要编码解决方案的量子比特索引。

    :retrun: 返回目标乘积的二进制表示。

    .. note::

        ``wires_m`` 使用的比特数量需要足够使用X基态编码 `m` 的二进制值。
        ``wires_k`` 使用的比特数量需要足够使用X基态编码 `k` 的二进制值。
        ``wires_solution`` 使用的比特数量需要足够使用X基态编码结果的二进制值。

    Example::

        import numpy as np
        from pyvqnet.qnn.vqc import QMachine,Samples, vqc_qft_mul
        wires_m = [0, 1, 2]           # qubits needed to encode m
        wires_k = [3, 4, 5]           # qubits needed to encode k
        wires_solution = [6, 7, 8, 9, 10]  # qubits needed to encode the solution
        
        dev = QMachine(len(wires_m) + len(wires_k) + len(wires_solution))

        vqc_qft_mul(dev,3, 7, wires_m, wires_k, wires_solution)


        ma = Samples(wires=wires_solution)
        y = ma(q_machine=dev)
        print(y)
        #[[1,0,1,0,1]]

VQC_FABLE
--------------------

.. py:class:: pyvqnet.qnn.vqc.VQC_FABLE(wires)

    使用快速近似块编码方法构建基于 VQC 的 QCircuit。对于特定结构的矩阵 [`arXiv:2205.00081 <https://arxiv.org/abs/2205.00081>`_], FABLE 方法可以简化块编码电路而不降低准确性。

    :param wires: 运算符作用的 qlist 索引。

    :return: 返回一个基于VQC的FABLE类实例。

    Example::

        from pyvqnet.qnn.vqc import VQC_FABLE
        from pyvqnet.qnn.vqc import QMachine
        from pyvqnet.dtype import float_dtype_to_complex_dtype
        import numpy as np
        from pyvqnet import QTensor
        
        A = QTensor(np.array([[0.1, 0.2 ], [0.3, 0.4 ]]) )
        qf = VQC_FABLE(list(range(3)))
        qm = QMachine(3,dtype=float_dtype_to_complex_dtype(A.dtype))
        qm.reset_states(1)
        z1 = qf(qm,A,0.001)
 
        """
        [[[[0.05     +0.j,0.15     +0.j],
        [0.05     +0.j,0.15     +0.j]],

        [[0.4974937+0.j,0.4769696+0.j],
        [0.4974937+0.j,0.4769696+0.j]]]]
        """


VQC_LCU
--------------------

.. py:class:: pyvqnet.qnn.vqc.VQC_LCU(wires=None, check_hermitian=True)

    使用线性组合单元 (LCU) 构建基于 VQC 的 QCircuit,`通过量子比特化进行哈密顿模拟 <https://arxiv.org/abs/1610.06546>`_。
    输入 dtype 可以是 kfloat32、kfloat64、kcomplex64、kcomplex128
    输入应为 Hermitian。

    :param wires: 运算符作用的 qlist 索引,可能需要辅助量子位。
    :param check_hermitian: 检查输入是否为 Hermitian,默认值:True。

    Example::

        from pyvqnet.qnn.vqc import VQC_LCU
        from pyvqnet.qnn.vqc import QMachine
        from pyvqnet.dtype import float_dtype_to_complex_dtype,kfloat64

        from pyvqnet import QTensor

        A = QTensor([[0.25,0,0,0.75],[0,-0.25,0.75,0],[0,0.75,0.25,0],[0.75,0,0,-0.25]],device=1001,dtype=kfloat64)
        qf = VQC_LCU(list(range(3)))
        qm = QMachine(3,dtype=float_dtype_to_complex_dtype(A.dtype))
        qm.reset_states(2)
        z1 = qf(qm,A)
        print(z1)
        """
        [[[[ 0.25     +0.j, 0.       +0.j],
        [ 0.       +0.j, 0.75     +0.j]],

        [[-0.4330127+0.j, 0.       +0.j],
        [ 0.       +0.j, 0.4330127+0.j]]],


        [[[ 0.25     +0.j, 0.       +0.j],
        [ 0.       +0.j, 0.75     +0.j]],

        [[-0.4330127+0.j, 0.       +0.j],
        [ 0.       +0.j, 0.4330127+0.j]]]]
        <QTensor [2, 2, 2, 2] DEV_CPU kcomplex128>
        """


VQC_QSVT
--------------------

.. py:class:: pyvqnet.qnn.vqc.VQC_QSVT(A, angles, wires)

    实现
    `quantum singular value transformation <https://arxiv.org/abs/1806.01838>`__ (QSVT) 线路.


    :param A: 要编码的一般 :math:`(n \times m)` 矩阵。
    :param angles:  构建QSVT多项式的偏移系数。
    :param wires: A 作用于的量子比特索引。

    Example::

        from pyvqnet import DEV_GPU
        from pyvqnet.qnn.vqc import QMachine,VQC_QSVT
        from pyvqnet.dtype import float_dtype_to_complex_dtype,kfloat64
        import numpy as np
        from pyvqnet import QTensor

        A = QTensor([[0.1, 0.2], [0.3, 0.4]])
        angles = QTensor([0.1, 0.2, 0.3])
        qm = QMachine(4,dtype=float_dtype_to_complex_dtype(A.dtype))
        qm.reset_states(1)
        qf = VQC_QSVT(A,angles,wires=[2,1,3])
        z1 = qf(qm)
        print(z1)
        """
        [[[[[ 0.9645935+0.2352667j,-0.0216623+0.0512362j],
        [-0.0062613+0.0308878j,-0.0199871+0.0985996j]],

        [[ 0.       +0.j       , 0.       +0.j       ],
            [ 0.       +0.j       , 0.       +0.j       ]]],


        [[[ 0.       +0.j       , 0.       +0.j       ],
            [ 0.       +0.j       , 0.       +0.j       ]],

        [[ 0.       +0.j       , 0.       +0.j       ],
            [ 0.       +0.j       , 0.       +0.j       ]]]]]
        """

量子机器学习模型接口
=====================


Quanvolution
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.qcnn.Quanvolution(params_shape, strides=(1, 1), kernel_initializer=quantum_uniform, machine_type: str = "cpu")


    基于《Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits》（https://arxiv.org/abs/1904.04767）实现的量子卷积，用变分量子电路替换经典的卷积滤波器，从而得到具有量子卷积滤波器的量子卷积神经网络。

    :param params_shape: 参数的形状，应为二维。
    :param strides: 切片窗口的步长，默认为 (1,1)。
    :param kernel_initializer: 参数的卷积核初始化器。
    :param machine_type: 机器类型字符串，默认为“cpu”。

    Example::

        from pyvqnet.qnn.qcnn import Quanvolution
        import pyvqnet.tensor as tensor
        qlayer = Quanvolution([4,2],(3,3))

        x = tensor.arange(1,25*25*3+1).reshape([3,1,25,25])

        y = qlayer(x)

        print(y.shape)

        y.backward()

        print(qlayer.m_para)
        print(qlayer.m_para.grad)
        #[3, 4, 8, 8]

        #[4.0270405,4.3587413,2.4935627,2.8155506,0.3314773,0.8889271,3.7357519, 0.9196261]
        #<Parameter [8] DEV_CPU kfloat32>

        #[ -0.2364242, -0.6942478, -8.445061 , -0.0558891, -0.       ,-49.498577 ,40.339344 , 40.339344 ]
        #<QTensor [8] DEV_CPU kfloat32>


QDRL
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.qdrl_vqc.QDRL(nq)


    基于《Data re-uploading for a universal quantum classifier》（https://arxiv.org/abs/1907.02085）实现的量子数据重上传(Quantum Data Re-upLoading,QDRL)算法是一种将量子电路与经典神经网络相结合的量子数据重上传模型。

    :param nq: 量子电路中所使用的量子比特（qubits的数量）。这决定了模型将要处理的量子系统的规模。


    Example::

        import numpy as np
        from pyvqnet.dtype import kfloat32
        from pyvqnet.qnn.qdrl_vqc import QDRL
        import pyvqnet.tensor as tensor

        # Set the number of quantum bits (qubits)
        nq = 1

        # Initialize the model
        model = QDRL(nq)

        # Create an example input (assume the input is a (batch_size, 3) shaped data)
        # Suppose we have a batch_size of 4 and each input has 3 features
        x_input = tensor.QTensor(np.random.randn(4, 3), dtype=kfloat32)

        # Pass the input through the model
        output = model(x_input)

        output.backward()

        # Output the result
        print("Model output:")
        print(output)


QGRU
------------------------------------------------------------

.. py:class:: pyvqnet.qnn.qgru.QGRU(para_num, num_of_qubits=4, input_size=100, hidden_size=100, batch_first=True)

    基于量子量子变分线路实现的GRU（门控循环单元），通过利用量子电路来进行状态更新和记忆保持。

    :param para_num: 量子电路中的参数数量。
    :param num_of_qubits: 量子比特的数量。
    :param input_size: 输入数据的特征维度。
    :param hidden_size: 隐藏单元的维度。
    :param batch_first:  输入的第一维是否为批次数量。

    Example::

        import numpy as np
        from pyvqnet.tensor import tensor
        from pyvqnet.qnn.qgru import QGRU
        from pyvqnet.dtype import kfloat32
        # Example usage
 
        # Set parameters
        para_num = 8
        num_of_qubits = 8
        input_size = 4
        hidden_size = 4
        batch_size = 1
        seq_length = 1
        # Create QGRU model
        qgru = QGRU(para_num, num_of_qubits, input_size, hidden_size, batch_first=True)

        # Create input data
        x = tensor.QTensor(np.random.randn(batch_size, seq_length, input_size), dtype=kfloat32)

        # Call the model
        output, h_t = qgru(x)
        output.backward()

        print("Output shape:", output.shape)  # Output shape
        print("h_t shape:", h_t.shape)  # Final hidden state shape




QLSTM
-------------------------------------------------------------

.. py:class:: pyvqnet.qnn.qlstm.QLSTM(para_num, num_of_qubits=4, input_size=100, hidden_size=100, batch_first=True)


    QLSTM (Quantum Long Short-Term Memory) 是一种结合了量子计算和经典LSTM的混合模型，旨在利用量子计算的并行性和经典LSTM的记忆能力来处理序列数据。

    :param para_num: 量子电路中的参数数量。
    :param num_of_qubits: 量子比特的数量。
    :param input_size: 输入数据的特征维度。
    :param hidden_size: 隐藏单元的维度。
    :param batch_first: 输入的第一维是否为批次数量。


    Example::

        import numpy as np
        from pyvqnet.tensor import tensor
        from pyvqnet.qnn.qlstm import QLSTM
        from pyvqnet.dtype import *
 
        para_num = 4
        num_of_qubits = 4
        input_size = 4
        hidden_size = 20
        batch_size = 3
        seq_length = 5
        qlstm = QLSTM(para_num, num_of_qubits, input_size, hidden_size, batch_first=True)
        x = tensor.QTensor(np.random.randn(batch_size, seq_length, input_size), dtype=kfloat32)

        output, (h_t, c_t) = qlstm(x)

        print("Output shape:", output.shape)
        print("h_t shape:", h_t.shape)
        print("c_t shape:", c_t.shape)

QMLPModel
--------------------------------------------------------------

.. py:class:: pyvqnet.qnn.qmlp.qmlp.QMLPModel(input_channels: int,output_channels: int,num_qubits: int, kernel: _size_type,stride: _size_type,padding: _padding_type = "valid",weight_initializer: Union[Callable, None] = None,bias_initializer: Union[Callable, None] = None,use_bias: bool = True,dtype: Union[int, None] = None)


    QMLPModel是基于《QMLP: An Error-Tolerant Nonlinear Quantum MLP Architecture using Parameterized Two-Qubit Gates》（https://arxiv.org/abs/2206.01345）实现的量子启发式神经网络，QMLPModel将量子电路与经典神经网络操作（如池化和全连接层）相结合。它旨在处理量子数据，并通过量子操作和经典层提取相关特征。

    :param input_channels: 输入特征的数量。
    :param output_channels: 输出特征的数量。
    :param num_qubits: 量子比特的数量。
    :param kernel: 平均池化窗口的大小。
    :param stride: 下采样的步长因子。
    :param padding: 填充方式，可选“valid”或“same”。
    :param weight_initializer: 权重初始化器，默认为正态分布。
    :param bias_initializer: 偏置初始化器，默认为零初始化。
    :param use_bias: 是否使用偏置，默认为True。
    :param dtype: 默认为None，使用默认数据类型。


    Example::

        import numpy as np
        from pyvqnet.tensor import tensor
        from pyvqnet.qnn.qmlp.qmlp import QMLPModel
        from pyvqnet.dtype import *

        input_channels = 16
        output_channels = 10
        num_qubits = 4
        kernel = (2, 2)
        stride = (2, 2)
        padding = "valid"
        batch_size = 8

        model = QMLPModel(input_channels=num_qubits,
                          output_channels=output_channels,
                          num_qubits=num_qubits,
                          kernel=kernel,
                          stride=stride,
                          padding=padding)

        x = tensor.QTensor(np.random.randn(batch_size, input_channels, 32, 32),dtype=kfloat32)

        output = model(x)

        print("Output shape:", output.shape)


QRLModel
-------------------------------------------------------------

.. py:class:: pyvqnet.qnn.qrl.QRLModel(num_qubits, n_layers=2)

    使用变分量子电路的量子深度强化学习模型,参考 :ref:`QDRL_DEMO` 中的线路实现。

    :param num_qubits: 量子电路中所使用的量子比特的数量。
    :param n_layers: 变分量子电路中的层数。


    Example::

        from pyvqnet.tensor import tensor, QTensor
        from pyvqnet.qnn.qrl import QRLModel

        num_qubits = 4
        model = QRLModel(num_qubits=num_qubits, n_layers=2)

        batch_size = 3
        x = QTensor([[1.1, 0.3, 1.2, 0.6], [0.2, 1.1, 0, 1.1], [1.3, 1.3, 0.3, 0.3]])
        output = model(x)
        output.backward()

        print("Model output:", output)


QRNN
--------------------------------------------------------------

.. py:class:: pyvqnet.qnn.qrnn.QRNN(para_num, num_of_qubits=4,input_size=100,hidden_size=100,batch_first=True)


    QRNN（Quantum Recurrent Neural Network）是一种量子循环神经网络，旨在处理序列数据并捕捉序列中的长期依赖关系。


    :param para_num: 量子电路中的参数数量。
    :param num_of_qubits: 量子比特的数量。
    :param input_size: 输入数据的特征维度。
    :param hidden_size: 隐藏单元的维度。
    :param batch_first: 输入的第一维是否为批次数量,默认为True。

    Example::

        from pyvqnet.dtype import kfloat32
        from pyvqnet.qnn.qrnn import QRNN
        from pyvqnet.tensor import tensor, QTensor
        import numpy as np


        para_num = 8
        num_of_qubits = 8
        input_size = 4
        hidden_size = 4
        batch_size = 1
        seq_length = 1
        qrnn = QRNN(para_num, num_of_qubits, input_size, hidden_size, batch_first=True)

        x = tensor.QTensor(np.random.randn(batch_size, seq_length, input_size), dtype=kfloat32)

        output, h_t = qrnn(x)

        print("Output shape:", output.shape)
        print("h_t shape:", h_t.shape)


TTOLayer
----------------------------------------------------------------

.. py:class:: pyvqnet.qnn.ttolayer.TTOLayer(inp_modes,out_modes,mat_ranks,biases_initializer=tensor.zeros)


    基于《Compressing deep neural networks by matrix product operators》（https://arxiv.org/abs/1904.06194）实现的TTOLayer对输入张量进行分解，从而实现对高维数据的高效表示。该层允许在秩约束条件下学习张量分解，相比于传统的全连接层，能够降低计算复杂度和内存使用。


    :param inp_modes: 输入张量的维度。
    :param out_modes: 输出张量的维度。
    :param mat_ranks: 张量分解中张量核（分解秩）的秩。
    :param biases_initializer: 偏置的初始化函数。

    Example::

        from pyvqnet.tensor import tensor
        import numpy as np
        from pyvqnet.qnn.ttolayer import TTOLayer
        from pyvqnet.dtype import kfloat32

        inp_modes = [4, 5]
        out_modes = [4, 5]
        mat_ranks = [1, 3, 1]
        tto_layer = TTOLayer(inp_modes, out_modes, mat_ranks)

        batch_size = 2
        len = 4
        embed_size = 5
        inp = tensor.QTensor(np.random.randn(batch_size, len, embed_size), dtype=kfloat32)

        output = tto_layer(inp)

        print("Input shape:", inp.shape)
        print("Output shape:", output.shape)





其他函数与类
=====================



QuantumLayerAdjoint
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.QuantumLayerAdjoint(vqc_module: pyvqnet.nn.Module, name="")

    使用伴随矩阵 (adjoint) 方式计算梯度的可自动微分的 QuantumLayer 层, 参考 `Efficient calculation of gradients in classical simulations of variational quantum algorithms <https://arxiv.org/abs/2009.02823>`_ 。

    伴随模式由本层在每次前向时自动打开: 前向开始时, ``vqc_module`` 使用的 ``QMachine`` 被切换到伴随模式 (此时 ``qm.is_adjoint()`` 为 ``True``), 线路中的门被记录而不是逐个执行; 前向结束后恢复为之前的模式 (``qm.is_adjoint()`` 为 ``False``)。调用 ``backward()`` 时, 从测量前的态矢出发反向扫描整个线路, 一次得到所有输入角度和所有可训练参数的梯度。``QMachine`` 本身不需要设置任何梯度模式。

    前向与反向默认优先使用分块 (tile) 伴随路径, 被拒绝时回退到逐门 (per-op) 伴随路径, 由环境变量 ``VQNET_ADJOINT_MODE`` 控制; 两条路径接纳的门、测量与数据布局见本节的 ``分块伴随路径`` 条目。``train()`` / ``eval()`` 的用法与其他 ``pyvqnet.nn.Module`` 相同, 是否处于训练模式不影响本层的前向与梯度计算。

    :param vqc_module: 一个仅使用 ``pyvqnet.qnn.vqc`` 下量子线路接口搭建的 `pyvqnet.nn.Module` 实例。
    :param name: 该层名字,默认为""。
    :return: 返回一个 QuantumLayerAdjoint 类实例。

    属性:

    * ``vqc_module``: 被包装的 VQC 模块。
    * ``qm``: ``vqc_module`` 使用的 ``QMachine``。
    * ``last_execution_impl``: 分块执行器驱动了前向之后为 ``"cpp"``; 前向由 Python 逐门路径执行时为 ``None``。该值不会被之后的逐门前向清除。
    * ``jac``: 雅可比矩阵, 形状为 ``[batch, 观测输出数, 可训练参数列数]``。列按线路中含参门被调用的顺序排列, 每个带角度的门占一列 (多参数门占多列), 需要梯度的输入列和可训练门的 ``Parameter`` 都计入; 同一个 ``Parameter`` 被多个门使用时每个门各占一列, 该参数的梯度是各列之和。第一次读取时才实际计算。

    .. note::

        测量必须使用支持伴随梯度的测量类: ``MeasureAll``、``Probability``、``QInfoMeasurement``。其他测量类 (例如 ``Samples``) 在前向时抛出 ``ValueError``。

        测量厄密算符的期望使用 ``QInfoMeasurement("expectation", obs=...)``: ``obs`` 中的矩阵可以是整个寄存器上的 ``2^n x 2^n`` 矩阵 (``wires`` 可以省略, 或者以任意顺序给出), 也可以是作用在 ``wires`` 上的小矩阵; 矩阵大小与 ``wires`` 不匹配时抛出 ``ValueError``。两种写法得到的前向值与梯度都和不使用本层时一致。

        逐门伴随路径支持由如下含参逻辑门 ``RX``, ``RY``, ``RZ``, ``PhaseShift``, ``RXX``, ``RYY``, ``RZZ``, ``RZX``, ``U1``, ``U2``, ``U3`` 以及其他不含参逻辑门构成的变分线路, 也支持可以分解为上述门的门 (例如 ``CRX``, ``CRY``, ``CRZ``, ``Toffoli``, ``CSWAP``, ``qft``, ``MulticontrolledX``)。三参数的 ``rot`` 以函数形式 (``pyvqnet.qnn.vqc.rot``) 调用时只能在分块路径下使用, 在逐门路径下反传时抛出 ``NotImplementedError``; ``Rot`` 模块形式在两条路径下都可以使用。

    .. py:method:: invalidate_execution_plans()

        丢弃已缓存的分块执行方案和上一次前向保存的反向上下文; 下一次前向重新记录线路。

    .. py:method:: invalidate_engine_plan()

        ``invalidate_execution_plans`` 的别名。

    Example::

        import pyvqnet
        from pyvqnet import tensor
        from pyvqnet.qnn.vqc import (
            CNOT, MeasureAll, QMachine, QuantumLayerAdjoint, RY, T, cnot, ry,
        )


        class QModel(pyvqnet.nn.Module):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()
                self.qm = QMachine(num_wires, dtype=dtype)
                self.ry0 = RY(trainable=True, wires=0, init_params=tensor.QTensor([[0.5]]))
                self.ry1 = RY(trainable=True, wires=1, init_params=tensor.QTensor([[-0.4]]))
                self.tlayer = T(wires=1)
                self.cnot = CNOT(wires=[0, 1])
                self.measure = MeasureAll(obs={"Z0": 1, "Z1": 1})

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])
                ry(self.qm, 0, x[:, 0:1])
                ry(self.qm, 1, x[:, 1:2])
                self.cnot(q_machine=self.qm)
                self.tlayer(q_machine=self.qm)
                self.ry0(q_machine=self.qm)
                self.ry1(q_machine=self.qm)
                return self.measure(q_machine=self.qm)


        input_x = tensor.QTensor([[0.1, 0.2], [0.3, -0.2], [0.5, 0.4], [0.7, -0.4]])
        input_x.requires_grad = True

        quantum_model = QModel(num_wires=3, dtype=pyvqnet.kcomplex64)
        adjoint_model = QuantumLayerAdjoint(quantum_model)
        adjoint_model.train()
        print("is_adjoint outside forward:", adjoint_model.qm.is_adjoint())

        batch_y = adjoint_model(input_x)
        print("last_execution_impl:", adjoint_model.last_execution_impl)
        batch_y.backward()
        print(batch_y)
        print(input_x.grad)
        print(quantum_model.ry0.params.grad)
        print(quantum_model.ry1.params.grad)


        # is_adjoint outside forward: False
        # last_execution_impl: cpp

        # [[1.8165869],
        #  [1.6742121],
        #  [1.5323746],
        #  [1.3331119]]
        # <QTensor [4, 1] DEV_CPU float32>

        # [[-0.2725032, 0.0408904],
        #  [-0.4351169, 0.3058294],
        #  [-0.9913   ,-0.2728508],
        #  [-0.9690847, 0.2434821]]
        # <QTensor [4, 2] DEV_CPU float32>

        # [[-1.6320289]]
        # <QTensor [1, 1] DEV_CPU float32>

        # [[1.3334608]]
        # <QTensor [1, 1] DEV_CPU float32>

    下面的例子在 ``QuantumLayerAdjoint`` 中用 ``QInfoMeasurement("expectation")`` 测量一个 4x4 厄密算符, 并与不使用本层的自动微分结果对比。

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.qnn.vqc import QInfoMeasurement, QuantumLayerAdjoint, QMachine, qcircuit
        from pyvqnet.tensor import QTensor


        class QModel(pyvqnet.nn.Module):
            """RX(theta) 与 CNOT 之后测量一个 4x4 厄密算符 H 的期望."""

            def __init__(self, obs):
                super().__init__()
                self.qm = QMachine(2, dtype=pyvqnet.kcomplex128)
                self.theta = QTensor(np.array([[0.7]], np.float64), requires_grad=True)
                self.measure = QInfoMeasurement("expectation", obs=obs)

            def forward(self, x):
                self.qm.reset_states(x.shape[0])
                qcircuit.ry(q_machine=self.qm, wires=0, params=self.theta)
                qcircuit.cnot(q_machine=self.qm, wires=[0, 1])
                return self.measure(q_machine=self.qm)


        H = np.array([[8, 4, 0, -6], [4, 0, 4, 0], [0, 4, 8, 0], [-6, 0, 0, 0]], np.complex128)
        H1 = np.array([[1, 0], [0, -1]], np.complex128)
        x = QTensor(np.zeros((2, 1)))

        for obs in ({"observables": QTensor(H)},
                    {"wires": (0, 1), "observables": QTensor(H)},
                    {"wires": (1, 0), "observables": QTensor(H)},
                    {"wires": (1,), "observables": QTensor(H1)}):
            text = {k: (tuple(v) if not isinstance(v, QTensor) else v.shape) for k, v in obs.items()}
            eager = QModel(obs)
            y_eager = eager(x)
            y_eager.sum().backward()
            layer = QModel(obs)
            y_tile = QuantumLayerAdjoint(layer)(x)
            y_tile.sum().backward()
            print(text, "value diff = %.1e, theta grad diff = %.1e" % (
                abs(float(y_tile.to_numpy().ravel()[0]) - float(y_eager.to_numpy().ravel()[0])),
                abs(float(layer.theta.grad.to_numpy().ravel()[0]) - float(eager.theta.grad.to_numpy().ravel()[0]))))

        mismatched = QModel({"wires": (0, 1), "observables": QTensor(H1)})
        try:
            QuantumLayerAdjoint(mismatched)(x)
        except ValueError as e:
            print(f"ValueError: {e}")


        # {'observables': [4, 4]} value diff = 0.0e+00, theta grad diff = 3.6e-15
        # {'wires': (0, 1), 'observables': [4, 4]} value diff = 0.0e+00, theta grad diff = 3.6e-15
        # {'wires': (1, 0), 'observables': [4, 4]} value diff = 1.8e-15, theta grad diff = 3.6e-15
        # {'wires': (1,), 'observables': [2, 2]} value diff = 0.0e+00, theta grad diff = 0.0e+00
        # ValueError: the observable is 2x2 but sits on 2 wire(s); pass the wires it acts on

    ``pyvqnet.qnn.vqc.QuantumAdjointLayer`` 是 ``QuantumLayerAdjoint`` 的别名。




分块伴随路径
---------------------------------------------------------------

``QuantumLayerAdjoint`` 的前向与反向默认优先使用分块 (tile) 伴随路径。该路径把记录下来的变分量子线路编译成若干个窗口 (window): 每个窗口是一段连续的量子比特, 由一次 C++ 核函数启动更新窗口内的门。前向按窗口作用整个线路并测量; 反向从测量前的态矢出发, 用一次核函数调用完成整个反向扫描, 得到雅可比矩阵 ``[batch, 观测输出数, 可训练参数列数]``。分块路径在 CPU 和 CUDA 的态矢模拟器上都可以运行。

某个步骤不满足分块路径的条件时称为被拒绝 (decline)。是否回退由环境变量 ``VQNET_ADJOINT_MODE`` 决定: 默认回退到逐门伴随路径, 前向结果与梯度不受影响; 仅分块模式下抛出异常。

分块路径接纳的线路:

* **数据类型**: 态矢为 ``complex64``、``complex128`` 或者 ``complex32`` 存储 (``QMachine(..., state_storage=pyvqnet.kcomplex32)``, 以 ``complex64`` 精度计算)。模拟器的 ``dtype`` 需要与态矢的存储类型一致 (``complex32`` 存储时 ``dtype`` 可以是 ``complex32`` 或 ``complex64``)。
* **窗口宽度**: CPU 上三种精度均为 13 个量子比特; CUDA 上 ``complex64`` 为 12 个, ``complex128`` 为 11 个, ``complex32`` 为 13 个。跨度超过窗口宽度的门中只有 ``SWAP`` 可以作为屏障门 (barrier, 单独作用在整个态矢上) 执行, 此时前向仍然使用分块路径而反向回退到逐门路径。
* **门** (函数形式和模块形式都可以):

  * 直接记录: ``RX``、``RY``、``RZ``、``PhaseShift``、``CRX``、``CRY``、``CRZ``、``CNOT``、``Hadamard``、``PauliX``、``PauliY``、``PauliZ``、``S``、``T``、``CZ``、``SWAP``、``Toffoli``、``I``、``CU1``。
  * 记录时被替换为上述单角度门的序列: ``U3``、``U2``、``Rot``、``CRot``、``RXX``、``RYY``、``RZZ``、``RZX``、``IsingXX``、``IsingYY``、``IsingZZ``、``IsingXY``、``MultiRZ``、``U1``、``X1``、``Y1``、``Z1``、``CY``、``CH``、``ISWAP``、``ControlledPhaseShift``; ``CR`` 被记录为 ``CU1``。
  * 任意酉矩阵门 ``unitary``: 作用在一个量子比特上时被替换为 ``U3``; 作用在两个量子比特上时以 4x4 矩阵记录 (反向使用其转置共轭); 作用在三个及以上量子比特时被拒绝。
  * 多比特受控 X 门 ``multicontrolledx`` / ``MultiControlledX`` (支持 ``control_values``): 两个控制比特时被替换为 ``Toffoli`` 序列, 三个及以上控制比特时被替换为 ``CU1`` 链。
  * ``cswap``、``vqc_ccz``、``qft``、``sdg``、``tdg``、``p``、``single_excitation``、``double_excitation``、``Projector`` 也可以在分块路径下运行。
  * 其他门 (例如自定义的门模块) 被拒绝。
* **测量**: 线路的最后必须恰好有一个测量, 并且是 ``MeasureAll``、``Probability`` 或 ``QInfoMeasurement``; 观测输出总数不超过 16 (``Probability`` 在 :math:`k` 个量子比特上有 :math:`2^k` 个输出)。
* **可训练参数**: 线路中至少有一个需要梯度的角度 (``requires_grad = True`` 的输入列, 或 ``trainable=True`` 的含参门)。初态必须是 :math:`|0\cdots 0\rangle`, 使用 ``init_states`` 设置其他初态时被拒绝。
* **角度布局**: 每个门的角度或者是整个批次共用一个值 (形状 ``[1, 1]`` 或 ``[1]``), 或者是每个样本一个值 (形状 ``[batch, 1]`` 或 ``[batch]``)。同一个按样本取值的角度列不能同时作为半角度门 (``RX``、``RY``、``RZ``、``CRX``、``CRY``、``CRZ`` 等) 和全角度门 (``PhaseShift``、``U1``、``CR``) 的参数, 例如不能把同一个 ``x[:, k:k+1]`` 同时传给 ``ry`` 和 ``phaseshift``。
* **不接纳的模拟器**: 带噪声信道的态矢模拟器 (密度矩阵模拟器由 ``VQNET_DM_MODE`` 选择执行路径), 以及使用量子比特重排 (``use_qr``)、张量网络或自定义通信器的模拟器。

下面的例子在 CPU 上训练一个 4 比特线路, 分别在 ``VQNET_ADJOINT_MODE`` 的三种取值下运行, 并与不使用 ``QuantumLayerAdjoint`` 的自动微分结果以及解析值对比梯度。

.. code-block::

    import os

    import numpy as np
    import pyvqnet
    from pyvqnet import tensor
    from pyvqnet.nn import Module, ModuleList
    from pyvqnet.optim import SGD
    from pyvqnet.qnn.vqc import (
        MeasureAll, QMachine, QuantumLayerAdjoint, RY, RZ, cnot, ry,
    )
    from pyvqnet.qnn.vqc.sv.common import tile_pass

    NUM_WIRES = 4


    class QModel(Module):
        """数据编码 RY(x) + CNOT 环 + 每个比特一个可训练 RZ 和一个可训练 RY,测量 <Z0>."""

        def __init__(self):
            super().__init__()
            self.qm = QMachine(NUM_WIRES, dtype=pyvqnet.kcomplex64)
            self.rz = ModuleList([RZ(trainable=True, wires=w, init_params=tensor.QTensor([[0.5]]))
                                  for w in range(NUM_WIRES)])
            self.ry = ModuleList([RY(trainable=True, wires=w, init_params=tensor.QTensor([[-0.4]]))
                                  for w in range(NUM_WIRES)])
            self.measure = MeasureAll(obs={"Z0": 1})

        def forward(self, x):
            self.qm.reset_states(x.shape[0])
            for w in range(NUM_WIRES):
                ry(self.qm, w, x[:, w:w + 1])                 # 数据编码角度: 输入的切片视图
            for w in range(NUM_WIRES):
                cnot(self.qm, [w, (w + 1) % NUM_WIRES])
            for w in range(NUM_WIRES):
                self.rz[w](q_machine=self.qm)
                self.ry[w](q_machine=self.qm)
            return self.measure(q_machine=self.qm)


    def make_data(batch=8):
        rng = np.random.RandomState(0)
        x = rng.uniform(-np.pi, np.pi, size=(batch, NUM_WIRES)).astype(np.float32)
        target = (np.cos(x[:, 0]) * 0.5).astype(np.float32)
        return tensor.QTensor(x), tensor.QTensor(target.reshape(batch, 1))


    def trainable_grads(model):
        return np.concatenate([g.params.grad.to_numpy().ravel()
                               for g in list(model.rz) + list(model.ry)])


    def train(mode, steps=30):
        os.environ["VQNET_ADJOINT_MODE"] = str(mode)
        tile_pass.TILE_PASS_STATS.reset()
        model = QModel()
        layer = QuantumLayerAdjoint(model)
        layer.train()
        opt = SGD(layer.parameters(), lr=0.1)
        x, target = make_data()
        losses = []
        for _ in range(steps):
            opt.zero_grad()
            y = layer(x)
            loss = tensor.mean((y - target) ** 2)
            loss.backward()
            opt.step()
            losses.append(float(loss.to_numpy()))
        print(f"VQNET_ADJOINT_MODE={mode}: loss {losses[0]:.6f} -> {losses[-1]:.6f}, "
              f"last_execution_impl={layer.last_execution_impl}, "
              f"tile_forward_calls={tile_pass.TILE_PASS_STATS.forward_calls}, "
              f"tile_backward_calls={tile_pass.TILE_PASS_STATS.backward_calls}")


    def adjoint_grads(mode):
        os.environ["VQNET_ADJOINT_MODE"] = str(mode)
        model = QModel()
        x, _ = make_data()
        x.requires_grad = True
        layer = QuantumLayerAdjoint(model)
        y = layer(x)
        tensor.sum(y).backward()
        return y.to_numpy().ravel(), x.grad.to_numpy().copy(), trainable_grads(model)


    # 1. 三种 VQNET_ADJOINT_MODE 下训练同一个线路
    for mode in (0, 1, 2):
        train(mode)

    # 2. 梯度检查: 不使用 QuantumLayerAdjoint 的自动微分结果作为参考
    model = QModel()
    x, _ = make_data()
    x.requires_grad = True
    y_auto = model(x)
    tensor.sum(y_auto).backward()
    ref = (y_auto.to_numpy().ravel(), x.grad.to_numpy().copy(), trainable_grads(model))
    for mode in (0, 1, 2):
        y, gx, gt = adjoint_grads(mode)
        print(f"VQNET_ADJOINT_MODE={mode}: max|dy|={np.abs(y - ref[0]).max():.2e}, "
              f"max|dx|={np.abs(gx - ref[1]).max():.2e}, "
              f"max|dtheta|={np.abs(gt - ref[2]).max():.2e}")


    # 3. 解析值检查: RY(theta)|0> 测量 <Z> = cos(theta), 梯度 = -sin(theta)
    class OneQubit(Module):
        def __init__(self):
            super().__init__()
            self.qm = QMachine(1, dtype=pyvqnet.kcomplex64)
            self.ry = RY(trainable=True, wires=0, init_params=tensor.QTensor([[0.3]]))
            self.measure = MeasureAll(obs={"Z0": 1})

        def forward(self, x):
            self.qm.reset_states(x.shape[0])
            self.ry(q_machine=self.qm)
            return self.measure(q_machine=self.qm)


    os.environ["VQNET_ADJOINT_MODE"] = "2"
    model = OneQubit()
    layer = QuantumLayerAdjoint(model)
    y = layer(tensor.QTensor([[0.0]]))
    y.backward()
    print(f"<Z0> = {float(y.to_numpy().ravel()[0]):.7f}, cos(0.3) = {np.cos(0.3):.7f}")
    print(f"d<Z0>/dtheta = {float(model.ry.params.grad.to_numpy().ravel()[0]):.7f}, "
          f"-sin(0.3) = {-np.sin(0.3):.7f}")


    # VQNET_ADJOINT_MODE=0: loss 0.398472 -> 0.311775, last_execution_impl=None, tile_forward_calls=0, tile_backward_calls=0
    # VQNET_ADJOINT_MODE=1: loss 0.398471 -> 0.311775, last_execution_impl=cpp, tile_forward_calls=30, tile_backward_calls=30
    # VQNET_ADJOINT_MODE=2: loss 0.398471 -> 0.311775, last_execution_impl=cpp, tile_forward_calls=30, tile_backward_calls=30
    # VQNET_ADJOINT_MODE=0: max|dy|=0.00e+00, max|dx|=2.68e-07, max|dtheta|=1.42e-07
    # VQNET_ADJOINT_MODE=1: max|dy|=1.19e-07, max|dx|=1.79e-07, max|dtheta|=2.38e-07
    # VQNET_ADJOINT_MODE=2: max|dy|=1.19e-07, max|dx|=1.79e-07, max|dtheta|=2.38e-07
    # <Z0> = 0.9553365, cos(0.3) = 0.9553365
    # d<Z0>/dtheta = -0.2955202, -sin(0.3) = -0.2955202


下面的线路包含多比特受控 X 门、``Toffoli``、``CRot``、单比特和两比特的任意酉矩阵门; 仅分块模式 (``VQNET_ADJOINT_MODE=2``) 下任何被拒绝的步骤都会抛出异常, 结果与自动微分一致。

.. code-block::

    import os

    import numpy as np
    import pyvqnet
    from pyvqnet import tensor
    from pyvqnet.nn import Module
    from pyvqnet.qnn.vqc import (
        CRot, MeasureAll, QMachine, QuantumLayerAdjoint, RY, cnot, hadamard,
        multicontrolledx, phaseshift, ry, toffoli, unitary,
    )

    rng = np.random.default_rng(4)
    u1 = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    U1 = tensor.QTensor(np.linalg.qr(u1)[0].astype(np.complex64))       # 单比特酉矩阵
    u2 = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    U2 = tensor.QTensor(np.linalg.qr(u2)[0].astype(np.complex64))       # 两比特酉矩阵
    x = tensor.QTensor(rng.uniform(-1.0, 1.0, size=(3, 5)).astype(np.float32))


    class QModel(Module):
        def __init__(self):
            super().__init__()
            self.qm = QMachine(5, dtype=pyvqnet.kcomplex64)
            self.theta = RY(trainable=True, wires=4, init_params=tensor.QTensor([[0.6]]))
            self.measure = MeasureAll(obs={"Z0": 1.0, "Z1": 0.5})

        def forward(self, x):
            self.qm.reset_states(x.shape[0])
            ry(self.qm, 4, x[:, 0:1])
            self.theta(q_machine=self.qm)                       # 可训练 Parameter
            cnot(self.qm, [4, 3])
            hadamard(self.qm, 2)
            cnot(self.qm, [2, 1])
            multicontrolledx(self.qm, [3, 2, 1, 0], control_values=[1, 0, 1])
            toffoli(self.qm, [0, 1, 2])
            CRot(wires=[2, 3], trainable=False)(params=x[:, 1:4], q_machine=self.qm)
            phaseshift(self.qm, 0, x[:, 4:5])                   # 全角度门使用单独的输入列
            unitary(self.qm, [1], U1)
            unitary(self.qm, [1, 2], U2, use_dagger=True)
            return self.measure(q_machine=self.qm)


    def run(model):
        x_in = tensor.QTensor(x.to_numpy().copy())
        x_in.requires_grad = True
        y = model(x_in)
        tensor.sum(y).backward()
        theta = model.vqc_module.theta if hasattr(model, "vqc_module") else model.theta
        return y.to_numpy().ravel(), x_in.grad.to_numpy().ravel(), theta.params.grad.to_numpy().ravel()


    # 参考: 不使用 QuantumLayerAdjoint 的自动微分
    ref_y, ref_dx, ref_dtheta = run(QModel())

    os.environ["VQNET_ADJOINT_MODE"] = "2"          # 仅分块: 任何被拒绝的步骤都会抛出异常
    layer = QuantumLayerAdjoint(QModel())
    y, dx, dtheta = run(layer)
    print("last_execution_impl:", layer.last_execution_impl)
    print("y =", np.round(y, 6))
    print("max|dy| =", float(np.abs(y - ref_y).max()))
    print("max|dx.grad| =", float(np.abs(dx - ref_dx).max()))
    print("max|dtheta.grad| =", float(np.abs(dtheta - ref_dtheta).max()))


    # last_execution_impl: cpp
    # y = [1.425533 1.401409 1.419964]
    # max|dy| = 4.76837158203125e-07
    # max|dx.grad| = 2.551823854446411e-07
    # max|dtheta.grad| = 1.471489667892456e-07


VQNET_ADJOINT_MODE
---------------------------------------------------------------

环境变量 ``VQNET_ADJOINT_MODE`` 选择 ``QuantumLayerAdjoint`` 使用的伴随执行路径。它在每次前向时读取, 可以在两次前向之间修改 ``os.environ``。

* ``0``: 仅使用逐门 (per-op) 伴随路径, 分块路径关闭。
* ``1`` (默认值,未设置或为空字符串): 优先使用分块路径; 某个步骤被分块路径拒绝时回退到逐门路径, 前向结果与梯度不受影响。
* ``2``: 仅使用分块路径; 前向或反向被拒绝时抛出 ``RuntimeError``, 信息以 ``VQNET_ADJOINT_MODE=2 but`` 开头并带有被拒绝的原因。
* 其他取值: 第一次前向时抛出 ``ValueError``。

默认模式下, 下列情况会使分块路径拒绝该步骤并回退到逐门路径: 门不在分块路径的门表内 (例如自定义的门模块、作用在三个及以上量子比特上的 ``unitary``)、观测输出超过 16 个、线路中没有需要梯度的角度、初态不是 :math:`|0\cdots 0\rangle`、角度布局无法表达、态矢模拟器带有噪声信道。线路中存在屏障门时只有反向回退到逐门路径。需要查看某个线路为什么没有走分块路径时, 可以临时设置 ``VQNET_ADJOINT_MODE=2`` 查看异常信息。

密度矩阵模拟器不读取该变量, 由 ``VQNET_DM_MODE`` 选择执行路径。

.. code-block::

    import os

    import numpy as np
    import pyvqnet
    from pyvqnet import tensor
    from pyvqnet.nn import Module
    from pyvqnet.qnn.vqc import (
        CNOT, MeasureAll, QMachine, QuantumLayerAdjoint, RY, unitary,
    )
    from pyvqnet.qnn.vqc.sv.common import tile_pass


    class QModel(Module):
        def __init__(self):
            super().__init__()
            self.qm = QMachine(3, dtype=pyvqnet.kcomplex64)
            self.measure = MeasureAll(obs={"Z0": 1})

        def forward(self, x):
            self.qm.reset_states(x.shape[0])
            RY(trainable=False, wires=2)(params=x[:, [0]], q_machine=self.qm)
            CNOT(wires=[2, 1])(q_machine=self.qm)
            RY(trainable=False, wires=0)(params=x[:, [1]], q_machine=self.qm)
            return self.measure(q_machine=self.qm)


    def run(model, mode):
        os.environ["VQNET_ADJOINT_MODE"] = str(mode)      # 每次前向时读取该环境变量
        tile_pass.TILE_PASS_STATS.reset()
        x = tensor.QTensor([[0.3, -0.2], [0.5, 0.1]])
        x.requires_grad = True
        layer = QuantumLayerAdjoint(model)
        y = layer(x)
        y.backward()
        print(f"VQNET_ADJOINT_MODE={mode}: y={np.round(y.to_numpy().ravel(), 6)}, "
              f"x.grad[:, 1]={np.round(x.grad.to_numpy()[:, 1], 6)}, "
              f"last_execution_impl={layer.last_execution_impl}, "
              f"tile_forward_calls={tile_pass.TILE_PASS_STATS.forward_calls}")


    for mode in (0, 1, 2):
        run(QModel(), mode)

    # 非法的取值在第一次前向时抛出 ValueError
    try:
        run(QModel(), 3)
    except ValueError as e:
        print(f"ValueError: {e}")


    # 分块路径不接纳的线路: 三比特的任意酉矩阵门
    class ThreeWireUnitary(Module):
        def __init__(self):
            super().__init__()
            self.qm = QMachine(3, dtype=pyvqnet.kcomplex64)
            self.mat = tensor.QTensor(np.eye(8, dtype=np.complex64) * 0.5)
            self.measure = MeasureAll(obs={"Z0": 1})

        def forward(self, x):
            self.qm.reset_states(x.shape[0])
            RY(trainable=False, wires=2)(params=x[:, [0]], q_machine=self.qm)
            CNOT(wires=[2, 1])(q_machine=self.qm)
            unitary(self.qm, [0, 1, 2], self.mat)
            RY(trainable=False, wires=0)(params=x[:, [1]], q_machine=self.qm)
            return self.measure(q_machine=self.qm)


    print("--- tile 路径拒绝该步骤 ---")
    run(ThreeWireUnitary(), 1)     # 回退到逐门 (per-op) 伴随路径, 结果仍然正确
    try:
        run(ThreeWireUnitary(), 2)  # 仅分块模式: 抛出异常并给出原因
    except RuntimeError as e:
        print(f"RuntimeError: {e}")


    # VQNET_ADJOINT_MODE=0: y=[0.980067 0.995004], x.grad[:, 1]=[ 0.198669 -0.099833], last_execution_impl=None, tile_forward_calls=0
    # VQNET_ADJOINT_MODE=1: y=[0.980067 0.995004], x.grad[:, 1]=[ 0.198669 -0.099833], last_execution_impl=cpp, tile_forward_calls=1
    # VQNET_ADJOINT_MODE=2: y=[0.980067 0.995004], x.grad[:, 1]=[ 0.198669 -0.099833], last_execution_impl=cpp, tile_forward_calls=1
    # ValueError: VQNET_ADJOINT_MODE must be 0 (per-op only), 1 (tile with fallback) or 2 (tile only), got '3'
    # --- tile 路径拒绝该步骤 ---
    # VQNET_ADJOINT_MODE=1: y=[0.245017 0.248751], x.grad[:, 1]=[ 0.049667 -0.024958], last_execution_impl=None, tile_forward_calls=0
    # RuntimeError: VQNET_ADJOINT_MODE=2 but the tile forward declined: the gate plan cannot be compiled for the tile bytecode (gate 2 'unitary' is outside the table)


分块方案的记录与重放
---------------------------------------------------------------

分块路径第一次运行一个线路时需要一次记录 (record) 过程: 以只记录不执行的方式运行一次用户的 ``forward``, 得到门序列、角度和测量, 再规划窗口并编译 (``TILE_PASS_STATS.plan_calls`` 加 1)。之后的训练步骤在满足下列条件时直接重放 (replay) 已记录的方案, 不再运行 ``forward``:

* 层输入 ``x`` 的批大小、数据类型、设备、形状、步长和 ``requires_grad`` 与记录时相同, 并且 ``forward`` 只接收 ``x`` 这一个参数。
* 每个门的角度是层输入 ``x`` 的切片视图 (``x[:, k:k+1]`` 或 ``x[:, k]``, 通过函数形式的门接口传入, 例如 ``ry(q_machine, wires, params)``), 或者是模块的 ``Parameter`` (``trainable=True`` 的含参门)。
* 没有设置 ``save_ir=True``, 前向过程中没有消耗随机数, 并且 ``VQNET_TILE_PLAN_CACHE`` 不为 ``0``。

不满足重放条件时, 每一步都重新记录线路。若线路结构 (门、量子比特、角度槽的布局)、批大小、精度和设备都不变, 窗口方案本身仍然从缓存中命中 (``plan_calls`` 为 0, ``plan_hits`` 加 1), 此时只重复记录过程。每个模拟器最多缓存 8 个方案, 窗口方案的进程级缓存最多 64 项。下列写法会使得每一步都重新记录线路 (结果不受影响):

* 角度写成 ``x[:, [k]]``: 列表索引得到的是副本, 不是输入的切片视图。
* 角度通过门模块的 ``params`` 参数传入, 例如 ``RY(trainable=False, wires=0)(params=x[:, 0:1], q_machine=qm)``。
* 角度由输入计算得到, 例如 ``x[:, 0:1] * 0.5``。
* 角度是常量, 例如 ``ry(qm, 0, 0.4)``, 包括 ``U2`` 被替换后引入的常量角度 :math:`\pi/2`。这种情况默认还会输出一次警告, 内容为 ``the tile plan cache cannot replay this layer: an angle view is computed outside the layer input and the Parameters, so every step records the circuit again. ...``。

批大小改变、调用 ``QuantumLayerAdjoint.invalidate_execution_plans()`` 或者设置了 ``VQNET_TILE_PLAN_CACHE=0`` 之后, 方案会被重新记录并重新规划。下面的例子展示记录、重放以及重新记录的时机。

.. code-block::

    import os

    import numpy as np
    import pyvqnet
    from pyvqnet import tensor
    from pyvqnet.nn import Module, ModuleList
    from pyvqnet.qnn.vqc import (
        CNOT, MeasureAll, QMachine, QuantumLayerAdjoint, RY, cnot, ry,
    )
    from pyvqnet.qnn.vqc.sv.common import tile_pass
    from pyvqnet.qnn.vqc.sv.common.tile_window_plan import window_plan_cache_info

    NUM_WIRES = 4


    class QModel(Module):
        def __init__(self):
            super().__init__()
            self.qm = QMachine(NUM_WIRES, dtype=pyvqnet.kcomplex64)
            self.rot = ModuleList([RY(trainable=True, wires=w, init_params=tensor.QTensor([[0.5]]))
                                   for w in range(NUM_WIRES)])
            self.measure = MeasureAll(obs={"Z0": 1})

        def forward(self, x):
            self.qm.reset_states(x.shape[0])
            for wire in range(NUM_WIRES):
                # 输入角度: 输入张量的切片视图, 分块路径据此重放已记录的方案
                ry(self.qm, wire, x[:, wire:wire + 1])
            for wire in range(NUM_WIRES - 1):
                cnot(self.qm, [wire, wire + 1])
            for wire in range(NUM_WIRES):
                self.rot[wire](q_machine=self.qm)     # 可训练参数: 模块 Parameter
            return self.measure(q_machine=self.qm)


    def step(layer, batch, seed):
        stats = tile_pass.TILE_PASS_STATS
        stats.reset()
        x = tensor.QTensor(np.random.RandomState(seed).randn(batch, NUM_WIRES).astype(np.float32))
        x.requires_grad = True
        y = layer(x)
        print(f"batch={batch}: y.shape={tuple(y.shape)}, "
              f"last_execution_impl={layer.last_execution_impl}, "
              f"last_execution_mode={layer.qm.last_execution_mode}")
        y.backward()
        print(f"           plan_calls={stats.plan_calls}, plan_hits={stats.plan_hits}, "
              f"forward_calls={stats.forward_calls}, backward_calls={stats.backward_calls}, "
              f"decline_reason={stats.decline_reason!r}")


    os.environ["VQNET_ADJOINT_MODE"] = "2"
    layer = QuantumLayerAdjoint(QModel())
    print("before any forward: last_execution_impl =", layer.last_execution_impl)
    for batch, seed in ((4, 0), (4, 1), (4, 2), (8, 3), (4, 4)):
        step(layer, batch, seed)
    print("window plan cache:", window_plan_cache_info())

    layer.invalidate_execution_plans()
    step(layer, 4, 5)


    # before any forward: last_execution_impl = None
    # batch=4: y.shape=(4, 1), last_execution_impl=cpp, last_execution_mode=tile
    #            plan_calls=1, plan_hits=0, forward_calls=1, backward_calls=1, decline_reason=''
    # batch=4: y.shape=(4, 1), last_execution_impl=cpp, last_execution_mode=tile
    #            plan_calls=0, plan_hits=0, forward_calls=1, backward_calls=1, decline_reason=''
    # batch=4: y.shape=(4, 1), last_execution_impl=cpp, last_execution_mode=tile
    #            plan_calls=0, plan_hits=0, forward_calls=1, backward_calls=1, decline_reason=''
    # batch=8: y.shape=(8, 1), last_execution_impl=cpp, last_execution_mode=tile
    #            plan_calls=1, plan_hits=0, forward_calls=1, backward_calls=1, decline_reason=''
    # batch=4: y.shape=(4, 1), last_execution_impl=cpp, last_execution_mode=tile
    #            plan_calls=0, plan_hits=1, forward_calls=1, backward_calls=1, decline_reason=''
    # window plan cache: {'size': 1, 'hits': 1, 'misses': 1, 'evictions': 0, 'max_size': 64}
    # batch=4: y.shape=(4, 1), last_execution_impl=cpp, last_execution_mode=tile
    #            plan_calls=1, plan_hits=0, forward_calls=1, backward_calls=1, decline_reason=''


VQNET_ADJOINT_ASSUME_LITERAL_ANGLES 与 VQNET_TILE_PLAN_CACHE
---------------------------------------------------------------

``VQNET_ADJOINT_ASSUME_LITERAL_ANGLES``: 设为 ``1`` 表示用户声明: 既不属于层输入也不属于 ``Parameter`` 的角度都是不变的常量 (字面量); 对于密度矩阵模拟器, 还声明线路中噪声信道的参数 (``p``、``gamma``) 每一步保持不变。设置后含有常量角度 (包括 ``U2``) 的线路也可以重放方案。默认 (未设置) 时, 这类角度无法与由输入计算得到的角度区分, 分块路径输出一次警告并让每一步都重新记录线路。由输入计算得到并且带梯度的角度 (例如 ``x[:, k:k+1] * 0.5``) 即使设置了该变量也不能重放。

``VQNET_TILE_PLAN_CACHE``: 设为 ``"0"`` 时关闭方案缓存, 每一步都重新记录线路并重新规划窗口; 其他取值 (默认) 打开缓存。

.. code-block::

    import os

    import numpy as np
    import pyvqnet
    from pyvqnet import tensor
    from pyvqnet.nn import Module
    from pyvqnet.qnn.vqc import MeasureAll, QMachine, QuantumLayerAdjoint, cnot, phaseshift, ry
    from pyvqnet.qnn.vqc.sv.common import tile_pass
    from pyvqnet.qnn.vqc.sv.common.tile_window_plan import clear_window_plan_cache, plan_cache_enabled


    class QModel(Module):
        """一个角度来自输入切片, 另一个角度是常量字面量."""

        def __init__(self):
            super().__init__()
            self.qm = QMachine(3, dtype=pyvqnet.kcomplex64)
            self.measure = MeasureAll(obs={"Z0": 1})

        def forward(self, x):
            self.qm.reset_states(x.shape[0])
            ry(self.qm, 2, x[:, 0:1])
            cnot(self.qm, [2, 1])
            phaseshift(self.qm, 1, 0.3)          # 既不是输入的切片, 也不是 Parameter
            ry(self.qm, 0, x[:, 1:2])
            return self.measure(q_machine=self.qm)


    def three_steps(label):
        clear_window_plan_cache()
        layer = QuantumLayerAdjoint(QModel())
        trace = []
        for step in range(3):
            tile_pass.TILE_PASS_STATS.reset()
            x = tensor.QTensor([[0.4, -0.2], [0.1, 0.5]])
            x.requires_grad = True
            y = layer(x)
            y.backward()
            stats = tile_pass.TILE_PASS_STATS
            trace.append((stats.plan_calls, stats.plan_hits))
        print(f"{label}: plan_cache_enabled={plan_cache_enabled()}, "
              f"(plan_calls, plan_hits) per step = {trace}")


    os.environ["VQNET_ADJOINT_MODE"] = "2"
    os.environ.pop("VQNET_ADJOINT_ASSUME_LITERAL_ANGLES", None)
    os.environ.pop("VQNET_TILE_PLAN_CACHE", None)
    three_steps("default switches")

    os.environ["VQNET_ADJOINT_ASSUME_LITERAL_ANGLES"] = "1"
    three_steps("VQNET_ADJOINT_ASSUME_LITERAL_ANGLES=1")
    os.environ.pop("VQNET_ADJOINT_ASSUME_LITERAL_ANGLES")

    os.environ["VQNET_TILE_PLAN_CACHE"] = "0"
    three_steps("VQNET_TILE_PLAN_CACHE=0")


    # default switches: plan_cache_enabled=True, (plan_calls, plan_hits) per step = [(1, 0), (0, 1), (0, 1)]
    # VQNET_ADJOINT_ASSUME_LITERAL_ANGLES=1: plan_cache_enabled=True, (plan_calls, plan_hits) per step = [(1, 0), (0, 0), (0, 0)]
    # VQNET_TILE_PLAN_CACHE=0: plan_cache_enabled=False, (plan_calls, plan_hits) per step = [(1, 0), (1, 0), (1, 0)]


观察分块路径的运行情况
---------------------------------------------------------------

``QuantumLayerAdjoint`` 和 ``QMachine`` 上的下列属性说明最近一次分块执行的情况:

* ``QuantumLayerAdjoint.last_execution_impl`` 与 ``qm.last_execution_impl``: 分块执行器驱动了前向之后为 ``"cpp"``; 初始值为 ``None``, 前向由 Python 逐门路径执行时保持为 ``None``。
* ``qm.last_execution_mode``: 分块前向之后为 ``"tile"``。分块路径运行过一次之前该属性不存在。

这两个属性记录的是最近一次分块前向, 之后的逐门前向不会清除它们。判断某一步是否走了分块路径, 以 ``TILE_PASS_STATS.reset()`` 之后 ``forward_calls`` 和 ``backward_calls`` 的增量为准。

.. py:data:: pyvqnet.qnn.vqc.sv.common.tile_pass.TILE_PASS_STATS

    分块路径的运行计数器 (进程内共用), ``reset()`` 将所有计数清零并把 ``decline_reason`` 置为空字符串。

    * ``forward_calls`` / ``backward_calls``: 分块前向 / 反向核函数运行的次数。
    * ``plan_calls`` / ``plan_hits``: 窗口方案的规划次数 / 缓存命中次数。
    * ``encode_calls``: 方案字节码的编码次数。
    * ``combine_calls`` / ``flat_calls``: 梯度汇总核函数的调用次数。
    * ``decline_reason``: 最近一次未被采用的子步骤的原因, 没有这样的子步骤时为空字符串。

    .. note::

        前向走了分块路径不保证反向也走分块路径: 线路中存在屏障门时反向回退到逐门路径, ``backward_calls`` 不增加。``decline_reason`` 在分块路径正常运行时也可能非空, 它记录的是可选子步骤未被采用的原因; 整个步骤被拒绝的原因在仅分块模式 (``VQNET_ADJOINT_MODE=2``) 的 ``RuntimeError`` 信息中给出。

.. py:function:: pyvqnet.qnn.vqc.sv.common.tile_window_plan.window_plan_cache_info()

    返回窗口方案缓存的统计信息: ``{"size", "hits", "misses", "evictions", "max_size"}``。

.. py:function:: pyvqnet.qnn.vqc.sv.common.tile_window_plan.clear_window_plan_cache()

    清空窗口方案缓存并清零其命中/未命中计数。


op_history_to_list
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.op_history_to_list(op_history, op_class_dict, use_param_with_autograd=False)

    把 ``QMachine`` 记录的 ``op_history`` 重建为算符 (``Operator``) 列表, 跳过其中的测量条目。伴随执行路径使用该函数从记录的线路重建门列表。

    :param op_history: ``QMachine.op_history`` 列表。
    :param op_class_dict: 门名称到算符类的字典, 例如 ``pyvqnet.backends.global_backend.op_class_dict()``。
    :param use_param_with_autograd: 为 ``True`` 时使用记录中带自动微分信息的参数 (``params_with_autograd``) 作为算符的参数, 否则使用记录的 ``params`` 并把算符设为不可训练。默认值: False。
    :return: 算符对象列表。

    Example::

        import pyvqnet
        from pyvqnet.qnn.vqc import CNOT, MeasureAll, QMachine, RY, op_history_to_list, rx, ry
        from pyvqnet.backends import global_backend
        from pyvqnet.tensor import QTensor

        qm = QMachine(2)
        qm.set_save_op_history_flag(True)
        qm.reset_states(2)
        ry(qm, 0, QTensor([[0.3], [0.5]]))
        CNOT(wires=[0, 1])(q_machine=qm)
        y = MeasureAll(obs={"Z1": 1})(q_machine=qm)

        print([entry["name"] for entry in qm.op_history])
        ops = op_history_to_list(qm.op_history, global_backend.op_class_dict(), True)
        for op in ops:
            print(type(op).__name__, op.wires, op.use_dagger, tuple(op.params.shape) if op.params is not None else None)


        # ['ry', 'cnot', 'MeasureAll']
        # RY (0,) False (2, 1)
        # CNOT (0, 1) False None

DataParallelVQCAdjointLayer
---------------------------------------------------------------

.. py:class:: pyvqnet.distributed.DataParallelVQCAdjointLayer(Comm_OP, vqc_module, name="", use_bucket_cpp_allreduce=True, bucket_bytes_cap=25*1024*1024, first_bucket_bytes_cap=1024*1024, find_unused_parameters=False, gradient_as_bucket_view=False)

    使用分布式数据并行训练变分量子线路并以伴随方式计算梯度的模块。该类继承自 ``QuantumLayerAdjoint``: ``vqc_module`` 是一个仅使用 ``pyvqnet.qnn.vqc`` 接口搭建的 VQC 模块, 每个进程持有一份相同的模型并处理自己的数据分片 (数据划分需要用户在调用 ``forward`` 之前完成); 反向传播时各进程的梯度通过 all-reduce 汇总 (MEAN 语义: 梯度先除以进程数再求和)。

    :param Comm_OP: 设置分布式环境的通信控制器。
    :param vqc_module: 带有 forward() 的 VQC模块,确保qmachine 已正确设置。
    :param name: 模块的名称。默认值为空字符串。
    :param use_bucket_cpp_allreduce: 是否使用 C++ 分桶 all-reduce 实现。当前必须为 True; 传 False 会抛出 NotImplementedError,因为逐参数的 all-reduce 已移除。默认值: True。
    :param bucket_bytes_cap: 梯度桶的最大字节数。默认值: 25*1024*1024。
    :param first_bucket_bytes_cap: 第一个梯度桶的最大字节数。默认值: 1024*1024。
    :param find_unused_parameters: 是否查找 autograd 图中未使用的参数。默认值: False。
    :param gradient_as_bucket_view: 是否将梯度暴露为通信桶的视图。默认值: False。
    :return: 可以计算量子电路的模块。

    Example::

        # 单进程运行: vqnetrun --nproc_per_node 1 python test.py
        # 多进程运行: vqnetrun --nproc_per_node 2 python test.py

        import pyvqnet
        from pyvqnet.device import DEV_GPU_0
        from pyvqnet.distributed import CommController, DataParallelVQCAdjointLayer
        from pyvqnet.nn import Module
        from pyvqnet.qnn.vqc import (
            CR, CZ, CNOT, Hadamard, I, MeasureAll, PauliX, PauliY, PauliZ, QMachine, RXX, RY,
            RZ, RZX, RX, RYY, RZZ, S, SWAP, T, Toffoli, U1, U2, U3, X1, Y1, Z1, iSWAP,
        )
        from pyvqnet.tensor import tensor

        bsize = 100


        class QModel(Module):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype)
                self.rx_layer = RX(trainable=False, wires=0)
                self.ry_layer = RY(trainable=False, wires=1)
                self.rz_layer = RZ(trainable=False, wires=1)
                self.u1 = U1(trainable=True, wires=[2])
                self.u2 = U2(trainable=True, wires=[3])
                self.u3 = U3(trainable=True, wires=[1])
                self.i = I(wires=[3])
                self.s = S(wires=[3])
                self.x1 = X1(wires=[3])
                self.y1 = Y1(wires=[3])
                self.z1 = Z1(wires=[3])
                self.x = PauliX(wires=[3])
                self.y = PauliY(wires=[3])
                self.z = PauliZ(wires=[3])
                self.swap = SWAP(wires=[2, 3])
                self.cz = CZ(wires=[2, 3])
                self.cr = CR(trainable=True, wires=[2, 3])
                self.rxx = RXX(trainable=True, wires=[2, 3])
                self.ryy = RYY(trainable=True, wires=[2, 3])
                self.rzz = RZZ(trainable=True, wires=[2, 3])
                self.rzx = RZX(trainable=False, wires=[2, 3])
                self.toffoli = Toffoli(wires=[2, 3, 4], use_dagger=True)

                self.h = Hadamard(wires=[1])

                self.iSWAP = iSWAP(wires=[0, 2])
                self.tlayer = T(wires=1)
                self.cnot = CNOT(wires=[0, 1])
                self.measure = MeasureAll(obs={'Z0': 2})

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])
                self.i(q_machine=self.qm)
                self.s(q_machine=self.qm)
                self.swap(q_machine=self.qm)
                self.cz(q_machine=self.qm)
                self.x(q_machine=self.qm)
                self.x1(q_machine=self.qm)
                self.y(q_machine=self.qm)
                self.y1(q_machine=self.qm)
                self.z(q_machine=self.qm)
                self.z1(q_machine=self.qm)
                self.rxx(q_machine=self.qm)
                self.ryy(q_machine=self.qm)
                self.rzz(q_machine=self.qm)
                self.rzx(q_machine=self.qm, params=x[:, [1]])

                self.u1(q_machine=self.qm)
                self.u2(q_machine=self.qm)
                self.u3(q_machine=self.qm)
                self.rx_layer(params=x[:, [0]], q_machine=self.qm)
                self.cnot(q_machine=self.qm)
                self.h(q_machine=self.qm)
                self.iSWAP(q_machine=self.qm)
                self.ry_layer(params=x[:, [1]], q_machine=self.qm)
                self.tlayer(q_machine=self.qm)
                self.rz_layer(params=x[:, [2]], q_machine=self.qm)
                self.toffoli(q_machine=self.qm)
                rlt = self.measure(q_machine=self.qm)

                return rlt


        pyvqnet.utils.set_random_seed(42)

        Comm_OP = CommController("nccl")

        input_x = tensor.QTensor([[0.1, 0.2, 0.3]])
        input_x = tensor.broadcast_to(input_x, [bsize, 3])
        input_x = input_x.toGPU(Comm_OP.get_local_rank() + DEV_GPU_0)
        input_x.requires_grad = True

        qunatum_model = QModel(num_wires=6, dtype=pyvqnet.kcomplex64)
        qunatum_model.toGPU(Comm_OP.get_local_rank() + DEV_GPU_0)

        l = DataParallelVQCAdjointLayer(
            Comm_OP,
            qunatum_model,
        )
        l.train()
        y = l(input_x)

        y.backward()
        print("rank", Comm_OP.get_local_rank(), "y.shape", tuple(y.shape),
              "x.grad.shape", tuple(input_x.grad.shape), "last_execution_impl", l.last_execution_impl)


        # rank 0 y.shape (100, 1) x.grad.shape (100, 3) last_execution_impl cpp



DataParallelVQCLayer
---------------------------------------------------------------

.. py:class:: pyvqnet.distributed.DataParallelVQCLayer(Comm_OP, vqc_module, name="", use_bucket_cpp_allreduce=True, bucket_bytes_cap=25*1024*1024, first_bucket_bytes_cap=1024*1024, find_unused_parameters=False, gradient_as_bucket_view=False)

    使用分布式数据并行训练变分量子线路并使用自动微分计算梯度的模块。``vqc_module`` 是一个带有 ``forward()`` 的 VQC 模块, 每个进程持有一份相同的模型并处理自己的数据分片 (数据划分需要用户在调用 ``forward`` 之前完成); 反向传播时各进程的梯度通过 all-reduce 汇总 (MEAN 语义: 梯度先除以进程数再求和)。

    :param Comm_OP: 设置分布式环境的通信控制器。
    :param vqc_module: 带有 forward() 的 VQC模块,确保qmachine 已正确设置。
    :param name: 模块的名称。默认值为空字符串。
    :param use_bucket_cpp_allreduce: 是否使用 C++ 分桶 all-reduce 实现。当前必须为 True; 传 False 会抛出 NotImplementedError,因为逐参数的 all-reduce 已移除。默认值: True。
    :param bucket_bytes_cap: 梯度桶的最大字节数。默认值: 25*1024*1024。
    :param first_bucket_bytes_cap: 第一个梯度桶的最大字节数。默认值: 1024*1024。
    :param find_unused_parameters: 是否查找 autograd 图中未使用的参数。默认值: False。
    :param gradient_as_bucket_view: 是否将梯度暴露为通信桶的视图。默认值: False。
    :return: 可以计算量子电路的模块。

    Example::

        # 单进程运行: vqnetrun --nproc_per_node 1 python xxx.py
        # 多进程运行: vqnetrun --nproc_per_node 2 python xxx.py

        import pyvqnet
        from pyvqnet.device import DEV_GPU_0
        from pyvqnet.distributed import CommController, DataParallelVQCLayer
        from pyvqnet.nn import Module
        from pyvqnet.qnn.vqc import MeasureAll, QMachine, cnot, rx, rz
        from pyvqnet.tensor import tensor


        class QModel(Module):
            def __init__(self, num_wires, num_layer, dtype):
                super(QModel, self).__init__()
                self.qm = QMachine(num_wires, dtype=dtype)
                self.measure = MeasureAll(obs={"Z0": 1})
                self.n = num_wires
                self.l = num_layer

            def forward(self, param, *args, **kwargs):
                n = self.n
                l = self.l
                qm = self.qm
                qm.reset_states(param.shape[0])
                for j in range(l):
                    cnot(qm, wires=[j, (j + 1) % l])
                    for i in range(n):
                        rx(qm, i, param[:, 3 * n * j + i])
                    for i in range(n):
                        rz(qm, i, param[:, 3 * n * j + i + n])
                    for i in range(n):
                        rx(qm, i, param[:, 3 * n * j + i + 2 * n])
                return self.measure(qm)


        n = 4
        b = 4
        l = 2

        input = tensor.ones([b, 3 * n * l])

        Comm = CommController("nccl")

        input = input.toGPU(Comm.get_local_rank() + DEV_GPU_0)
        input.requires_grad = True
        qunatum_model = QModel(num_wires=n, num_layer=l, dtype=pyvqnet.kcomplex64)
        qunatum_model.toGPU(Comm.get_local_rank() + DEV_GPU_0)

        layer = DataParallelVQCLayer(
            Comm,
            qunatum_model,
        )
        y = layer(input)
        y.backward()
        print("rank", Comm.get_local_rank(), "y.shape", tuple(y.shape), "x.grad.shape", tuple(input.grad.shape))


        # rank 0 y.shape (4, 1) x.grad.shape (4, 24)


vqc_to_originir_list
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.vqc_to_originir_list(vqc_model: pyvqnet.nn.Module)

    将 VQNet vqc 模块转换为 `originIR <https://qpanda-tutorial.readthedocs.io/zh/latest/QProgToOriginIR.html#id2>`_ 。

    vqc_model 应在此函数之前运行前向函数以获取输入数据。
    如果输入数据是批量数据。 对于每个输入,它将返回多个 IR 字符串。

    :param vqc_model: VQNet vqc 模块,应该先向前运行。

    :return: originIR 字符串或 originIR 字符串列表。

    Example::

        import pyvqnet
        import pyvqnet.tensor as tensor
        from pyvqnet.qnn.vqc import *
        from pyvqnet.nn import Module
        from pyvqnet.utils import set_random_seed
        set_random_seed(42)
        class QModel(Module):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype,save_ir=True)
                self.rx_layer = RX(trainable=False, wires=0)
                self.ry_layer = RY(trainable=False, wires=1)
                self.rz_layer = RZ(trainable=False, wires=1)
                self.u1 = U1(trainable=True,wires=[2])
                self.u2 = U2(trainable=True,wires=[3])
                self.u3 = U3(trainable=True,wires=[1])
                self.i = I(wires=[3])
                self.s = S(wires=[3])
                self.x1 = X1(wires=[3])
                self.y1 = Y1(wires=[3])
                self.z1 = Z1(wires=[3])
                self.x = PauliX(wires=[3])
                self.y = PauliY(wires=[3])
                self.z = PauliZ(wires=[3])
                self.swap = SWAP(wires=[2,3])
                self.cz = CZ(wires=[2,3])
                self.cr = CR(trainable=True,wires=[2,3])
                self.rxx = RXX(trainable=True,wires=[2,3])
                self.rzz = RYY(trainable=True,wires=[2,3])
                self.ryy = RZZ(trainable=True,wires=[2,3])
                self.rzx = RZX(trainable=False, wires=[2,3])
                self.toffoli = Toffoli(wires=[2,3,4],use_dagger=True)

                self.h =Hadamard(wires=[1])
                self.rot = VQC_HardwareEfficientAnsatz(6, ["rx", "RY", "rz"],
                                                    entangle_gate="cnot",
                                                    entangle_rules="linear",
                                                    depth=5)
                
                self.iSWAP = iSWAP(wires=[0,2])
                self.tlayer = T(wires=1)
                self.cnot = CNOT(wires=[0, 1])
                self.measure = MeasureAll(obs = {
                    "X1":1
                })

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])
                self.i(q_machine=self.qm)
                self.s(q_machine=self.qm)
                self.swap(q_machine=self.qm)
                self.cz(q_machine=self.qm)
                self.x(q_machine=self.qm)
                self.x1(q_machine=self.qm)
                self.y(q_machine=self.qm)
                self.y1(q_machine=self.qm)
                self.z(q_machine=self.qm)
                self.z1(q_machine=self.qm)
                self.ryy(q_machine=self.qm)
                self.rxx(q_machine=self.qm)
                self.rzz(q_machine=self.qm)
                self.rzx(q_machine=self.qm,params = x[:,[1]])
                self.cr(q_machine=self.qm)
                self.u1(q_machine=self.qm)
                self.u2(q_machine=self.qm)
                self.u3(q_machine=self.qm)
                self.rx_layer(params = x[:,[0]], q_machine=self.qm)
                self.cnot(q_machine=self.qm)
                self.h(q_machine=self.qm)
                self.iSWAP(q_machine=self.qm)
                self.ry_layer(params = x[:,[1]], q_machine=self.qm)
                self.tlayer(q_machine=self.qm)
                self.rz_layer(params = x[:,[2]], q_machine=self.qm)
                self.toffoli(q_machine=self.qm)
                rlt = self.measure(q_machine=self.qm)

                return rlt
            

        input_x = tensor.QTensor([[0.1, 0.2, 0.3]])

        input_x = tensor.broadcast_to(input_x,[2,3])

        input_x.requires_grad = True

        qunatum_model = QModel(num_wires=6, dtype=pyvqnet.kcomplex64)

        batch_y = qunatum_model(input_x)
        batch_y.backward()
        ll = vqc_to_originir_list(qunatum_model)
        from pyqpanda3.intermediate_compiler import convert_originir_string_to_qprog

        for l in ll :
            print(l)

            prog = convert_originir_string_to_qprog(l)
            print(prog)


originir_to_vqc
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.originir_to_vqc(originir, tmp="code_tmp.py", verbose=False)

    将 originIR 解析为 vqc 模型代码。
    代码创建一个没有 `Measure` 的变分量子线路 `pyvqnet.nn.Module` ,返回是量子态的态矢形式,如 [b,2,...,2]。
    该函数将在“./origin_ir_gen_code/” + tmp +“.py”中生成一个定义对应VQNet模型的代码文件。
    
    :param originir: 原始IR。
    :param tmp: 代码文件名,默认 ``code_tmp.py``。
    :param verbose: 如果显示生成代码,默认 = False
    :return:
        生成可运行代码。

    Example::

        from pyvqnet.qnn.vqc import originir_to_vqc
        ss = "QINIT 3\nCREG 3\nH q[1]"
        
        Z = originir_to_vqc(ss,verbose=True)

        exec(Z)
        m =Exported_Model()
        print(m(2))

        # from pyvqnet.nn import Module
        # from pyvqnet.tensor import QTensor
        # from pyvqnet.qnn.vqc import *
        # class Exported_Model(Module):
        #         def __init__(self, name=""):
        #                 super().__init__(name)

        #                 self.q_machine = QMachine(num_wires=3)
        #                 self.H_0 = Hadamard(wires=1, use_dagger = False)

        #         def forward(self, x, *args, **kwargs):
        #                 x = self.H_0(q_machine=self.q_machine)
        #                 return self.q_machine.states

        # [[[[0.7071068+0.j 0.       +0.j]
        #    [0.7071068+0.j 0.       +0.j]]

        #   [[0.       +0.j 0.       +0.j]
        #    [0.       +0.j 0.       +0.j]]]]


model_summary
---------------------------------------------------------------

.. py:function:: pyvqnet.model_summary(vqc_module)

    打印在 vqc_module 中注册的经典层和量子门运算符的信息。
    
    :param vqc_module: vqc 模块
    :return:
         摘要字符串


    Example::

        from pyvqnet.qnn.vqc import QMachine, RX, RY, CNOT, PauliX, qmatrix, PauliZ,MeasureAll
        from pyvqnet.tensor import QTensor, tensor,kcomplex64
        import pyvqnet
        from pyvqnet.nn import LSTM,Linear
        from pyvqnet import model_summary
        class QModel(pyvqnet.nn.Module):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype)
                self.rx_layer1 = RX(trainable=True,
                                    wires=1,
                                    init_params=tensor.QTensor([0.5]))
                self.ry_layer2 = RY(trainable=True,
                                    wires=0,
                                    init_params=tensor.QTensor([-0.5]))
                self.xlayer = PauliX(wires=0)
                self.cnot = CNOT(wires=[0, 1])
                self.measure = MeasureAll(obs = PauliZ)
                self.linear = Linear(24,2)
                self.lstm =LSTM(23,5)
            def forward(self, x, *args, **kwargs):
                return super().forward(x, *args, **kwargs)
        Z = QModel(4,kcomplex64)

        print(model_summary(Z))
        # ###################QModel Summary#######################

        # classic layers: {'Linear': 1, 'LSTM': 1}
        # total classic parameters: 650

        # =========================================
        # qubits num: 4
        # gates: {'RX': 1, 'RY': 1, 'PauliX': 1, 'CNOT': 1}
        # total quantum gates: 4
        # total quantum parameter gates: 2
        # total quantum parameters: 2
        # #########################################################


QNG
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.QNG(qmodel, stepsize=0.01, momentum=0)

    `量子自然梯度法(Quantum Nature Gradient) <https://arxiv.org/abs/1909.02108>`_ 借鉴经典自然梯度法的概念 `Amari (1998) <https://www.mitpressjournals.org/doi/abs/10.1162/089976698300017746>`__ ,
    我们改为将优化问题视为给定输入的可能输出值的概率分布(即,最大似然估计),则更好的方法是在分布
    空间中执行梯度下降,它相对于参数化是无量纲和不变的. 因此,无论参数化如何,每个优化步骤总是会为每个参数选择最佳步长。
    在量子机器学习任务中,量子态空间拥有一个独特的不变度量张量,称为 Fubini-Study 度量张量 :math:`g_{ij}`。
    该张量将量子线路参数空间中的最速下降转换为分布空间中的最速下降。
    量子自然梯度的公式如下:

    .. math:: \theta_{t+1} = \theta_t + momentum(x^{(t)} - x^{(t-1)}) - \eta g^{+}(\theta_t)\nabla \mathcal{L}(\theta),

    其中 :math:`g^{+}` 是伪逆。

    ``wrapper_calculate_qng`` 是需要加到待计算量子自然梯度的模型的forward函数的装饰器。仅对模型注册的 `Parameter` 类型的参数优化。

    Fubini-Study 度量按线路的含参层分块计算, 再通过门角度对各叶子参数 (输入和 ``Parameter``) 的 classical Jacobian 映射到叶子参数上; 由输入缩放得到的角度 (例如 ``x[:, [k]] * 0.5``) 按同样的方式得到自然梯度。

    :param qmodel: 量子变分线路模型,需要使用 `wrapper_calculate_qng` 作为forward函数的装饰器,并且至少含有一个注册的 ``Parameter`` (``trainable=True`` 的含参门),否则构造时抛出 ``ValueError``。
    :param stepsize: 梯度下降法的步长,默认0.01。
    :param momentum: 动量，默认0.

    .. note::

        仅在非批处理数据上进行了测试,``step()`` 要求 ``qm.batch_size == 1``。
        仅支持纯变分量子电路。
        step() 将更新输入和参数的梯度。
        step() 仅会更新模型参数的数值,输入 ``x`` 的数值保持不变。
        度量张量奇异时 (例如作用在 :math:`|0\rangle` 上的 ``RZ`` 门,其生成元在该态下方差为 0) ``step()`` 抛出 ``numpy.linalg.LinAlgError``。

    .. py:method:: step(x, g=None)

        运行一次前向,计算量子自然梯度并用 ``stepsize`` 更新模型参数。

        :param x: 输入参数,需要有 ``requires_grad = True``,形状为 ``[x]`` 或 ``[1, x]``; 没有输入参数时为 ``None``。
        :param g: 输出的梯度,默认为 None (即全 1)。

    .. py:method:: zero_grad()

        清零优化器中参数的梯度。

    Example::

        from pyvqnet.qnn.vqc import QMachine, RX, RY, RZ, CNOT, rz, PauliX, qmatrix, PauliZ, Probability, rx, ry, MeasureAll, U2
        from pyvqnet.tensor import QTensor, tensor
        import pyvqnet
        import numpy as np

        from pyvqnet.qnn.vqc import wrapper_calculate_qng

        class QModel(pyvqnet.nn.Module):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype)
                self.rz_layer1 = RZ(trainable=False, wires=0)
                self.rz_layer2 = RZ(trainable=False, wires=1)
                self.u2_layer1 = U2(trainable=False, wires=0)
                self.l_train1 = RY(trainable=True, wires=1)
                self.l_train1.params.init_from_tensor(
                    QTensor([333], dtype=pyvqnet.kfloat32))
                self.l_train2 = RX(trainable=True, wires=2)
                self.l_train2.params.init_from_tensor(
                    QTensor([4444], dtype=pyvqnet.kfloat32))
                self.xlayer = PauliX(wires=0)
                self.cnot01 = CNOT(wires=[0, 1])
                self.cnot12 = CNOT(wires=[1, 2])
                self.measure = MeasureAll(obs={'Y0': 1})

            @wrapper_calculate_qng
            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])

                ry(q_machine=self.qm, wires=0, params=np.pi / 4)
                ry(q_machine=self.qm, wires=1, params=np.pi / 3)
                ry(q_machine=self.qm, wires=2, params=np.pi / 7)
                self.rz_layer1(q_machine=self.qm, params=x[:, [0]])
                self.rz_layer2(q_machine=self.qm, params=x[:, [1]])

                self.u2_layer1(q_machine=self.qm, params=x[:, [4, 5]])  #

                self.cnot01(q_machine=self.qm)
                self.cnot12(q_machine=self.qm)
                ry(q_machine=self.qm, wires=0, params=np.pi / 7)
                ry(q_machine=self.qm, wires=1, params=x[:, [2]])
                rx(q_machine=self.qm, wires=2, params=x[:, [3]])
                rz(q_machine=self.qm, wires=1, params=x[:, [3]])
                ry(q_machine=self.qm, wires=0, params=np.pi / 7)
                rz(q_machine=self.qm, wires=1, params=x[:, [3]])

                self.cnot01(q_machine=self.qm)
                self.cnot12(q_machine=self.qm)
                rlt = self.measure(q_machine=self.qm)
                return rlt

        qmodel = QModel(3, pyvqnet.kcomplex64)

        x = QTensor([[1111.0, 2222, 333,444, 55, 666]], requires_grad=True)

        qng = pyvqnet.qnn.vqc.QNG(qmodel,0.01)
        qng.step(x)

        print(x)
        print(x.grad)
        print(qmodel.l_train1.params)
        print(qmodel.l_train2.params)



        # [[1111.,2222., 333., 444.,  55., 666.]]
        # <QTensor [1, 6] DEV_CPU float32>

        # [[ 0.4306372, 0.0076731,-0.9832919, 1.0491865, 1.1636074, 0.0925422]]
        # <QTensor [1, 6] DEV_CPU float32>

        # [[333.]]
        # <Parameter [1, 1] DEV_CPU float32>

        # [[4444.]]
        # <Parameter [1, 1] DEV_CPU float32>

    下面的例子中线路只有一个可训练的 ``RY`` 门: :math:`C = \langle Z_0\rangle = \cos\theta`, 度规 :math:`G = 1/4`, 一步量子自然梯度更新为 :math:`\theta' = \theta + 4\eta\sin\theta`。

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.qnn.vqc import MeasureAll, QMachine, RY, wrapper_calculate_qng
        from pyvqnet.tensor import QTensor, tensor


        class SingleRY(pyvqnet.nn.Module):
            """RY(theta)|0> 测量 <Z0>: 代价 C = cos(theta), 度规 G = 1/4."""

            def __init__(self):
                super().__init__()
                self.qm = QMachine(1, dtype=pyvqnet.kcomplex64)
                self.ry = RY(trainable=True, wires=0, init_params=tensor.QTensor([[0.3]]))
                self.measure = MeasureAll(obs={"Z0": 1})

            @wrapper_calculate_qng
            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(1)
                self.ry(q_machine=self.qm)
                return self.measure(q_machine=self.qm)


        model = SingleRY()
        qng = pyvqnet.qnn.vqc.QNG(model, stepsize=0.1)
        qng.step(None)                         # 没有输入参数时传入 None
        theta = float(model.ry.params.to_numpy().ravel()[0])
        print(f"theta after one QNG step: {theta:.7f}")
        print(f"closed form 0.3 + 4 * 0.1 * sin(0.3): {0.3 + 4 * 0.1 * np.sin(0.3):.7f}")


        # theta after one QNG step: 0.4182081
        # closed form 0.3 + 4 * 0.1 * sin(0.3): 0.4182081

.. _precompile_execution:

预编译执行
---------------------------------------------------------------

在 native 后端 (``pyvqnet`` 或 ``pyvqnet-ad``) 上, 态矢位于 CUDA 设备的 ``pyvqnet.qnn.vqc.QMachine`` 会自动把一次前向中的门记录下来而不是逐个执行: ``reset_states`` 打开一次记录, 线路末尾的 ``MeasureAll`` 触发执行。记录下的门序列先由 C++ 融合 pass 处理, 再交给融合执行器以一次调用运行, 记为 ``"cxx"``; 门或测量不能交给融合执行器时, 记录下的门按记录顺序通过逐门接口重放, 记为 ``"replay"``; 没有打开记录的线路逐门执行, 记为 ``"eager"``。预编译不需要用户开启: 满足条件的模拟器自动使用, 结果与梯度和逐门执行一致。

.. note::

    ``"cxx"`` 路径需要同时满足:

    * 模拟器的态矢在 CUDA 设备上。CPU 上的态矢总是 ``"eager"``。
    * 线路处于普通执行阶段: 没有调用 ``set_just_defined(True)`` 或 ``set_save_op_history_flag(True)``, 没有 ``save_ir=True``、量子比特重排 (``use_qr``)、张量网络、自定义通信器, 也没有打开 ``set_enable_decompose(True)``。``QuantumLayerAdjoint`` 的逐门前向不使用融合执行器。
    * 线路以一个 ``MeasureAll`` 结束, 其 ``obs`` 是 Pauli 项的字典或列表, 系数为常数, 并且不含全为 ``I`` 的项。``MeasureAll(obs=PauliZ)`` 这种算符类的形式不能交给融合执行器。
    * 记录的门都在融合执行器的门表中。

    不满足后两条时按 ``"replay"`` 执行。测量之前读取态矢 (例如访问 ``qm.states``) 也会结束记录并逐门执行这些门。

**门表**: 融合执行器直接运行 ``RX``、``RY``、``RZ``、``CRX``、``CRY``、``CRZ``、``CNOT``、``Hadamard``、``PauliX``、``PauliY``、``PauliZ``、``SWAP`` 以及作用在一个量子比特上的 ``unitary``。其他门 (例如 ``PhaseShift``、``U3``、``S``、``T``、``Toffoli``、作用在两个量子比特上的 ``unitary``) 使整个线路按 ``"replay"`` 执行。

**融合规则**: ``"cxx"`` 路径依次使用 ``cancel_inverses`` (抵消相邻的自逆门对: ``H``、``X``、``Y``、``Z``、``CNOT``、``CZ``、``SWAP``)、``merge_rotations`` (合并相邻的同类型旋转门) 和 ``promote_fixed_rotations`` (把相邻的固定 Pauli 门并入同轴的旋转门, 例如 ``X`` 并入 ``RX``, 差一个全局相位)。``"replay"`` 路径另外使用 ``fuse_disjoint_layers`` 把不相交的门分组成融合核函数。记录的门超过 256 个时跳过分组规则。融合后的方案按记录下来的线路结构缓存 (最多 64 项), 结构相同的下一次前向直接复用。

**关闭预编译**: 把态矢放在 CPU 上; 或者在调用 ``reset_states`` 之前调用 ``qm.set_save_op_history_flag(True)``; 或者设置模块开关 ``pyvqnet.qnn.vqc.sv.common.cxx_runner.CXX_RUNNER_AVAILABLE = False``。

**执行模式切换**: ``reset_states`` 打开记录之后再切换执行模式 (例如 ``set_just_defined(True)``、``set_save_op_history_flag(True)``、进入 ``QuantumLayerAdjoint`` 的伴随前向) 会丢弃该记录, 并且下一个门调用抛出 ``RuntimeError: cxx_runner: a recording was open when another execution mode started ...``。需要在这些模式内调用 ``reset_states``。另外, 已安装的噪声策略不能作用于融合执行器记录的线路, ``pyvqnet.qnn.vqc.common.noise_policy.validate_execution_config`` 在门或测量入口抛出 ``RuntimeError``。

.. py:function:: pyvqnet.qnn.vqc.sv.common.cxx_runner.last_execution(q_machine)

    返回模拟器最近一次运行完的线路所走的执行路径: ``"cxx"``、``"replay"`` 或 ``"eager"``。

    :param q_machine: ``QMachine`` 实例。
    :return: 字符串。

.. py:function:: pyvqnet.qnn.vqc.sv.common.cxx_runner.fused_plan_cache_stats()

    返回融合方案缓存的计数: ``{"hits", "misses", "uncacheable", "evictions", "size", "max_size"}``。

.. py:function:: pyvqnet.qnn.vqc.sv.common.cxx_runner.reset_fused_plan_cache()

    清空融合方案缓存并清零计数。

下面的例子需要在 GPU 上运行: 同一线路在关闭和打开融合执行器时的结果与梯度一致; 含有 ``CZ`` 的线路按 ``"replay"`` 执行。

.. code-block::

    import numpy as np
    import pyvqnet
    from pyvqnet import DEV_GPU_0
    from pyvqnet.nn import Module
    from pyvqnet.qnn.vqc import CNOT, CZ, MeasureAll, QMachine, RX, RY, RZ
    from pyvqnet.qnn.vqc.sv.common import cxx_runner
    from pyvqnet.tensor import QTensor, tensor


    class Circuit(Module):
        """RY / RX 层 + CNOT + 一个 RZ,测量两个输出列."""

        def __init__(self, with_cz=False):
            super().__init__()
            self.qm = QMachine(3, dtype=pyvqnet.kcomplex64)
            self.ry0 = RY(trainable=True, wires=0, init_params=tensor.QTensor([[0.4]]))
            self.rx1 = RX(trainable=True, wires=1, init_params=tensor.QTensor([[1.1]]))
            self.rz2 = RZ(trainable=True, wires=2, init_params=tensor.QTensor([[0.9]]))
            self.with_cz = with_cz

        def forward(self, x):
            self.qm.reset_states(x.shape[0])
            self.ry0(q_machine=self.qm)
            self.rx1(q_machine=self.qm)
            CNOT(wires=[0, 1])(q_machine=self.qm)
            if self.with_cz:
                CZ(wires=[1, 2])(q_machine=self.qm)       # CZ 不在融合执行器的门表内
            self.rz2(q_machine=self.qm)
            return MeasureAll(obs={"Z0": 1.0, "Z1": 0.5})(q_machine=self.qm)


    def run(model, x_data, device):
        x = QTensor(x_data, requires_grad=True, device=device)
        y = model(x)
        y_data = y.to_numpy().ravel().copy()
        tensor.sum(y).backward()
        grads = {name: p.grad.to_numpy().ravel().copy() for name, p in model.named_parameters()}
        return y_data, grads


    x_data = np.array([[0.3, 0.1], [0.5, 0.2]], np.float32)

    cpu = Circuit()
    run(cpu, x_data, pyvqnet.DEV_CPU)
    print("CPU         :", cxx_runner.last_execution(cpu.qm))

    eager = Circuit().toGPU(DEV_GPU_0)
    cxx = Circuit().toGPU(DEV_GPU_0)
    cxx_runner.CXX_RUNNER_AVAILABLE = False            # 关闭融合执行器: 逐门执行
    y_eager, g_eager = run(eager, x_data, DEV_GPU_0)
    print("GPU, off    :", cxx_runner.last_execution(eager.qm))
    cxx_runner.CXX_RUNNER_AVAILABLE = True             # 默认: 打开
    y_cxx, g_cxx = run(cxx, x_data, DEV_GPU_0)
    print("GPU, on     :", cxx_runner.last_execution(cxx.qm))
    run(cxx, x_data, DEV_GPU_0)                        # 相同结构再运行一次: 命中融合方案缓存
    print("fused plan cache:", cxx_runner.fused_plan_cache_stats())
    print("max|dy| =", float(np.abs(y_cxx - y_eager).max()))
    for name in g_cxx:
        print(f"max|d{name}| =", float(np.abs(g_cxx[name] - g_eager[name]).max()))

    refused = Circuit(with_cz=True).toGPU(DEV_GPU_0)
    y_refused, _ = run(refused, x_data, DEV_GPU_0)
    print("GPU, with CZ:", cxx_runner.last_execution(refused.qm))
    print("max|dy| =", float(np.abs(y_refused - y_eager).max()))


    # CPU         : eager
    # GPU, off    : eager
    # GPU, on     : cxx
    # fused plan cache: {'hits': 1, 'misses': 1, 'uncacheable': 0, 'evictions': 0, 'size': 1, 'max_size': 64}
    # max|dy| = 0.0
    # max|dry0.params| = 0.0
    # max|drx1.params| = 0.0
    # max|drz2.params| = 0.0
    # GPU, with CZ: replay
    # max|dy| = 0.0


PrecompileNotSupported
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.PrecompileNotSupported

    ``RuntimeError`` 的子类。张量网络模块 ``pyvqnet.qnn.vqc.tn.native.TNQModule`` 和 ``pyvqnet.qnn.vqc.tn.torch.TNQModule`` 在第一次调用时先记录线路再编译 (``use_jit=True``)。线路无法被记录时抛出该异常, 不会回退到逐门执行; 每次调用都会重新尝试记录并再次抛出。

    无法被记录的情况:

    * 门参数不是注册的 ``Parameter`` (赋值为模块属性的 ``trainable=True`` 的门或 ``Parameter``)、直接传入的输入列 (``x[k]``) 或常量。由模块输入计算得到的角度 (``x * x``、缩放或重组的特征) 不能被记录。
    * 使用 ``VQC_IQPEmbedding`` 这类内部角度由输入计算得到的接口。
    * JIT 编译失败并且关闭 JIT 重试也失败, 信息以 ``TN JIT recording failed`` 开头。

    遇到该异常时, 可以把该线路放在不带 JIT 的 ``TNQMachine`` 上逐门执行, 或者把计算得到的角度作为额外的输入列传给 ``TNQModule``。

    Example::

        # 没有可用 GPU 时设置 JAX_PLATFORMS=cpu, 避免 jax 的 CUDA 插件输出报错信息

        import numpy as np
        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc import PrecompileNotSupported
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn.native import Probability, RX, TNQMachine, TNQModule, VQC_IQPEmbedding


        class Recordable(TNQModule):
            """角度是直接的输入列 (x[0]) 和模块 Parameter: 可以被记录并编译."""

            def __init__(self):
                super().__init__()
                self.device = TNQMachine(2, dtype=pyvqnet.kcomplex64)
                self.g = RX(True, init_params=0.3, wires=0)      # 注册为模块属性的可训练门
                self.measure = Probability(wires=[0])

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(x.shape[0])
                RX(trainable=False, wires=1)(params=x[0], q_machine=self.device)
                self.g(q_machine=self.device)
                return self.measure(q_machine=self.device)


        class NotRecordable(TNQModule):
            """VQC_IQPEmbedding 的两比特门角度由输入计算得到: 无法被记录."""

            def __init__(self):
                super().__init__()
                self.device = TNQMachine(3, dtype=pyvqnet.kcomplex64)
                self.measure = Probability(wires=[0, 2])

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(x.shape[0])
                VQC_IQPEmbedding(x, q_machine=self.device)
                return self.measure(q_machine=self.device)


        module = Recordable()
        x = QTensor(np.array([[0.4, 0.5]], np.float32), requires_grad=True)
        y = module(x)
        print("compiled:", module._jit_compiled, "use_jit:", module.use_jit, "y:", np.round(y.to_numpy(), 6))

        try:
            NotRecordable()(QTensor(np.array([[0.4, 0.5, 0.6]], np.float32), requires_grad=True))
        except PrecompileNotSupported as e:
            print(f"PrecompileNotSupported: {e}")


        # compiled: True use_jit: True y: [[0.977668 0.022332]]
        # PrecompileNotSupported: TN interface 'VQC_IQPEmbedding' requires eager execution: its gate angles are computed from the module input (input-derived arithmetic), which the JIT recorder cannot replay. A TNQModule records its circuit before running it, and this interface cannot be recorded: keep it on a bare TNQMachine (eager per-gate execution), or hand the gate angles over as input columns, module parameters or plain constants.


QNSPSAOptimizer
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.QNSPSAOptimizer(stepsize=1e-3,regularization=1e-3,finite_diff_step=1e-2,resamplings=1,blocking=True,history_length=5,seed=None)

    ​​量子自然 SPSA (QNSPSA) 优化器

    一个用于量子电路的二阶随机优化器，结合了梯度下降和 Fubini-Study 度量张量信息。

    使用对称扰动进行梯度估计​​（类似 SPSA）：

    .. math::

        \widehat{\nabla f}(\mathbf{x}) \approx \frac{f(\mathbf{x}+\epsilon \mathbf{h})-f(\mathbf{x}-\epsilon \mathbf{h})}{2\epsilon}

    通过状态重叠测量计算 Fubini-Study 度量：

    .. math::

        \widehat{\mathbf{g}}(\mathbf{x}) \approx \frac{\delta F}{8\epsilon^2}(\mathbf{h}_1\mathbf{h}_2^\intercal + \mathbf{h}_2\mathbf{h}_1^\intercal)

    .. math::

        \delta F = F(\mathbf{x}+\epsilon\mathbf{h}_1+\epsilon\mathbf{h}_2) - F(\mathbf{x}+\epsilon\mathbf{h}_1) - F(\mathbf{x}-\epsilon\mathbf{h}_1+\epsilon\mathbf{h}_2) + F(\mathbf{x}-\epsilon\mathbf{h}_1)

    其中 δF 测量四个电路的重叠差异评价。

    ​​更新规则​​：

    .. math::

        \mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} - \eta \widehat{\mathbf{g}}^{-1}(\mathbf{x}^{(t)})\widehat{\nabla f}(\mathbf{x}^{(t)})

    :param stepsize: 用户自定义学习率超参数 :math:`\eta` （默认值：1e-3）
    :param regularization: 用于 Fubini-Study 度量张量的正则化项 :math:`\beta` ，用于数值稳定性（默认值：1e-3）
    :param finite_diff_step: 用于计算有限差分梯度和 Fubini-Study 度量张量的步长 :math:`\epsilon` （默认值：1e-2）
    :param resamplings: 每次参数更新的平均样本数（默认值：1）
    :param blocking: 当为 True 时，仅接受使损失值不大于前一次损失值加容差的更新（有助于收敛）（默认值：True）
    :param history_length: 当 ``blocking`` 为 True 时，容差设置为前几个 ``history_length`` 成本值的平均值（默认值：5）
    :param seed: 随机采样的种子（默认值：None）

    模型的 ``forward`` 只能使用 ``MeasureAll`` 作为测量。

    支持批量数据: 传入的可训练参数的第一维是数据批量大小, 线路的 ``reset_states`` 按该批量运行。此时每个样本的测量必须恰好给出一个代价值 (一个观测量的期望), 优化器最小化的代价是各样本代价值的平均值, Fubini-Study 度量的估计也按样本取平均; 测量对每个样本给出多个输出时抛出 ``ValueError``。

    .. py:method:: step(qmodel, *args, **kwargs)

        使用优化器的更新可训练参数一次。

        :param qmodel: 可训练的量子模型
        :param args: 用于 qmodel 的可变长度可训练 qtensor。``requires_grad = True`` 的参数被更新并作为新的 ``QTensor`` 返回, 其他参数原样返回; 只有一个参数时直接返回该参数, 否则返回参数列表。
        :param kwargs: 用于 qmodel 的可变长度关键字参数。

        :return: 更新后的参数。

    .. py:method:: step_and_cost(cost, *args, **kwargs)

        更新一次参数并返回当前代价。

        :param cost: 可训练的量子模型
        :param args: 用于 qmodel 的可变长度可训练 qtensor。
        :param kwargs: 用于 qmodel 的可变长度关键字参数。

        :return: ``(更新后的参数, 代价)``。``blocking`` 为 ``True`` 时, 代价是更新之前参数对应的代价值 (批量数据时为各样本代价的平均值); 为 ``False`` 时是更新之后参数的代价。

    .. py:method:: execute(tapes, qm, batch)

        静态方法: 在批大小 ``batch`` 下重放记录下的线路 ``tapes`` (门模块列表, 最后一个元素是测量模块) 并返回测量结果。调用结束后恢复 ``qm`` 的态矢、批大小和模式标志。

        :param tapes: 由 ``construct_tape`` 记录的列表: 门模块, 末尾是测量模块。
        :param qm: 线路使用的 ``QMachine``。
        :param batch: 重放时的批大小。
        :return: 测量结果, ``QTensor``。

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc import rx, ry, cry, QMachine, MeasureAll, QModule, QNSPSAOptimizer


        class QModuleDemo(QModule):
            """单样本: RX(theta0) 与 CRY(theta1),测量 <Z1 Z0>."""

            def __init__(self, name=""):
                super().__init__(name)
                self.qm = QMachine(2)
                self.ma = MeasureAll({"Z1 Z0": 1})

            def forward(self, params):
                qm = self.qm
                qm.reset_states(1)
                rx(qm, 0, params[0])
                cry(qm, [0, 1], params[1])
                return self.ma(qm)


        qmd = QModuleDemo()
        params = QTensor([0.37454012, 0.95071431])
        params.requires_grad = True
        opt = QNSPSAOptimizer(stepsize=5e-2, seed=1)
        for i in range(51):
            params = opt.step(qmd, params)
            loss = qmd(params)
            if i % 10 == 0:
                print(f"Step {i}: cost = {loss.to_numpy()}")


        class QModuleBatch(QModule):
            """批量数据: 每个样本先作用 RY(x_b), 再作用 RY(theta); 代价是各样本 <Z0> 的平均值."""

            def __init__(self, data, name=""):
                super().__init__(name)
                self.qm = QMachine(1)
                self.ma = MeasureAll({"Z0": 1})
                self.data = data

            def forward(self, params):
                qm = self.qm
                qm.reset_states(self.data.shape[0])
                ry(qm, 0, self.data)
                ry(qm, 0, params[0])
                return self.ma(qm)


        data = QTensor([0.2, 0.65, 1.1])
        qmb = QModuleBatch(data)
        theta = QTensor([0.4])
        theta.requires_grad = True
        opt = QNSPSAOptimizer(stepsize=5e-2, seed=7)
        for i in range(31):
            theta = opt.step(qmb, theta)
            if i % 10 == 0:
                print(f"Step {i}: cost = {qmb(theta).mean().to_numpy()}")
        theta, loss = opt.step_and_cost(qmb, theta)
        print("theta =", theta.to_numpy(), ", cost =", loss.to_numpy())
        print("argmin of mean_b cos(x_b + theta) =", np.pi - float(data.to_numpy().mean()))


        # Step 0: cost = [[0.9346176]]
        # Step 10: cost = [[0.43794498]]
        # Step 20: cost = [[-0.59825623]]
        # Step 30: cost = [[-0.9935878]]
        # Step 40: cost = [[-0.99988145]]
        # Step 50: cost = [[-0.99999183]]
        # Step 0: cost = 0.41116321086883545
        # Step 10: cost = -0.6304031014442444
        # Step 20: cost = -0.9210363030433655
        # Step 30: cost = -0.9332655072212219
        # theta = [2.4681888] , cost = -0.9332655
        # argmin of mean_b cos(x_b + theta) = 2.4915926178270063


可训练性诊断
---------------------------------------------------------------

``pyvqnet.qnn.vqc.diagnostics`` 在一系列随机初始化和噪声设置上测量梯度的方差, 用来判断变分量子线路是否会出现贫瘠高地 (barren plateau)。每个配置点使用参数平移 (位移 :math:`\pm\pi/2`) 计算每个旋转门参数的梯度, 代价是全局磁化强度 :math:`(1/n)\sum_i \langle Z_i\rangle`; 带噪声时对每个种子的若干条 Monte-Carlo 轨迹取平均。

.. py:data:: pyvqnet.qnn.vqc.diagnostics.ANSATZE

    内置拟设的名称元组: ``("hardware_efficient", "strongly_entangling")``。``"hardware_efficient"`` 每层是一排 ``RY`` 加一个 ``CNOT`` 环; ``"strongly_entangling"`` 每层是 ``RY`` 加 ``CNOT`` 环, 再接 ``RZ`` 加 ``CZ`` 环。每层的参数个数分别为 ``n_qubits`` 和 ``2 * n_qubits``。

.. py:data:: pyvqnet.qnn.vqc.diagnostics.RISK_SLOPE_HIGH

    高风险的最大宽度斜率, ``-3.0``。

.. py:data:: pyvqnet.qnn.vqc.diagnostics.RISK_VARIANCE_HIGH

    高风险的最大梯度方差, ``1e-3``。

.. py:data:: pyvqnet.qnn.vqc.diagnostics.RISK_SLOPE_MEDIUM

    中等风险的最大宽度斜率, ``-1.5``。

.. py:data:: pyvqnet.qnn.vqc.diagnostics.RISK_VARIANCE_MEDIUM

    中等风险的最大梯度方差, ``1e-2``。

    ``summarize_trainability`` 用这四个阈值标记风险: ``log(平均梯度方差)`` 对 ``log(量子比特数)`` 的最小二乘斜率不超过 ``RISK_SLOPE_HIGH`` 或者平均梯度方差小于 ``RISK_VARIANCE_HIGH`` 时为 ``"high"``; 斜率不超过 ``RISK_SLOPE_MEDIUM`` 或者平均梯度方差小于 ``RISK_VARIANCE_MEDIUM`` 时为 ``"medium"``; 否则为 ``"low"``。

.. py:class:: pyvqnet.qnn.vqc.diagnostics.TrainabilityConfig(ansatz, n_qubits, depth, noise, strength, trajectories, seeds, dtype=pyvqnet.kcomplex64, device=pyvqnet.DEV_CPU, state_storage=None)

    扫描矩阵中的一个配置点 (不可变的 dataclass)。种子同时决定参数初始化的取值和噪声的随机序列。

    :param ansatz: 拟设名称, 必须是 ``ANSATZE`` 之一。
    :param n_qubits: 量子比特数, 正整数。
    :param depth: 层数, 正整数。
    :param noise: 噪声信道, ``None`` 或 ``"depolarizing"``。带噪声时每层之后在每个量子比特上作用去极化信道。
    :param strength: 噪声强度, 取值范围 ``[0, 1]``; ``noise`` 为 ``None`` 时必须为 0。
    :param trajectories: 每个参数点的 Monte-Carlo 轨迹数, 正整数。
    :param seeds: 随机种子的元组, 不能为空。
    :param dtype: ``pyvqnet.kcomplex64`` 或 ``pyvqnet.kcomplex128``。默认值: ``pyvqnet.kcomplex64``。
    :param device: 运行设备。默认值: ``pyvqnet.DEV_CPU``。
    :param state_storage: 态矢的存储类型, 可选 ``pyvqnet.kcomplex32``; 默认值: ``None``。
    :return: 配置对象。取值不合法时抛出 ``ValueError``。

.. py:function:: pyvqnet.qnn.vqc.diagnostics.quick_trainability_config(ansatz, seeds=(0, 1, 2))

    返回一个用于快速检查的配置: 4 个量子比特、深度 1、无噪声、每个种子 32 条轨迹。

    :param ansatz: 拟设名称, ``ANSATZE`` 之一。
    :param seeds: 随机种子的元组。默认值: ``(0, 1, 2)``。
    :return: ``TrainabilityConfig`` 对象。

.. py:function:: pyvqnet.qnn.vqc.diagnostics.gradient_samples(cfg, samples_per_seed=1)

    计算一个配置的参数平移梯度样本。每个种子用 ``numpy.random.default_rng(seed)`` 在 :math:`[0, 2\pi)` 内均匀抽取参数点, 每个参数点有 :math:`P` 个参数 (:math:`P` 为该拟设的参数个数), 参数点数向上取整为 ``samples_per_seed / P``。

    :param cfg: ``TrainabilityConfig`` 对象。
    :param samples_per_seed: 每个种子需要的梯度系数个数, 正整数。默认值: 1。
    :return: ``numpy`` 数组, 形状 ``[参数点数 * P, 种子数]``。

.. py:function:: pyvqnet.qnn.vqc.diagnostics.scan_trainability(configs, samples_per_seed=1, bootstrap_replicates=200)

    依次运行每个配置, 返回逐种子的梯度方差记录。

    :param configs: ``TrainabilityConfig`` 的可迭代对象, 不能为空。
    :param samples_per_seed: 每个种子的梯度系数个数, 会向上取整到完整的参数点。默认值: 1。
    :param bootstrap_replicates: 计算方差相对标准误 (``rse``) 的 bootstrap 重抽样次数。默认值: 200。
    :return: 字典列表, 每个 (配置, 种子) 一项。字典的键为 ``ansatz``、``n_qubits``、``depth``、``noise``、``strength``、``trajectories``、``seed``、``n_parameters``、``n_points``、``n_coefficients``、``samples_per_seed``、``grad_variance``、``grad_variance_mean_marginal``、``grad_kurtosis_excess``、``effective_n``、``rse``、``mean_gradient``; 配置设置了 ``state_storage`` 时另有 ``state_storage``。

.. py:function:: pyvqnet.qnn.vqc.diagnostics.summarize_trainability(rows)

    把逐种子的记录汇总为每个配置一组的统计量和改进建议。

    :param rows: ``scan_trainability`` 返回的列表, 每项必须含有 ``grad_variance``。
    :return: 字典, 键为 ``variance_means`` (所有记录的 ``grad_variance``)、``configs``、``recommendations`` 和 ``risk_thresholds`` (四个风险阈值)。``configs`` 的每一项含有 ``ansatz``、``n_qubits``、``depth``、``noise``、``strength``、``trajectories``、``seeds``、``seed_variances``、``mean_grad_variance``、``std_grad_variance``、``ci95``、``ci95_half_width``、``rse_mean``、``rse_max``、``n_points_mean``、``effective_n_mean``、``grad_variance_mean_marginal_mean``、``grad_kurtosis_excess_mean``、``slope_vs_width`` 和 ``risk``。``recommendations`` 的每一项含有 ``id``、``action`` 和 ``evidence``, ``id`` 为 ``cap_width``、``cap_depth``、``noise_budget``、``start_config``、``seed_precision`` 或 ``risk_summary``。

.. py:function:: pyvqnet.qnn.vqc.diagnostics.write_report(results, path)

    把 ``summarize_trainability`` 的结果 (可以附加 ``rows`` 和 ``metadata``) 写入 JSON 文件 ``path``, 并在同目录写入同名的 Markdown 报告 (扩展名为 ``.md``)。

    :param results: 字典, 必须含有非空的 ``configs`` 和 ``recommendations``。
    :param path: 输出的 JSON 文件路径, 目录不存在时会创建。
    :return: 无返回值。

.. code-block::

    from dataclasses import replace

    from pyvqnet.qnn.vqc.diagnostics import (
        ANSATZE,
        RISK_SLOPE_HIGH,
        RISK_VARIANCE_HIGH,
        TrainabilityConfig,
        gradient_samples,
        quick_trainability_config,
        scan_trainability,
        summarize_trainability,
        write_report,
    )

    # 1. 快速配置: 4 个比特, 深度 1, 无噪声, 32 条轨迹
    print(ANSATZE)
    print(quick_trainability_config("hardware_efficient", seeds=(0, 1)))

    # 2. 扫描矩阵: 2 个拟设 x 2 种宽度, 每个拟设 2 个随机种子
    configs = []
    for ansatz in ANSATZE:
        for width in (4, 6):
            configs.append(replace(quick_trainability_config(ansatz, seeds=(0, 1)), n_qubits=width))
    rows = scan_trainability(configs, samples_per_seed=64)
    print(len(rows), sorted(rows[0]))

    # 3. 汇总每个配置并给出建议
    summary = summarize_trainability(rows)
    for config in summary["configs"]:
        print(config["ansatz"], config["n_qubits"], f"{config['mean_grad_variance']:.6g}",
              config["risk"], f"{config['slope_vs_width']:.3f}")
    for recommendation in summary["recommendations"]:
        print(recommendation["id"], ":", recommendation["action"])

    # 4. 写出 JSON 与 Markdown 报告
    results = dict(summary)
    results["rows"] = rows
    write_report(results, "out/trainability_report.json")

    # 5. 单独获取梯度样本: 形状 [n_points * P, 种子数]
    samples = gradient_samples(quick_trainability_config("hardware_efficient", seeds=(0,)))
    print(samples.shape)

    # 6. 自定义配置: 带去极化噪声
    noisy = TrainabilityConfig(ansatz="hardware_efficient", n_qubits=4, depth=2,
                               noise="depolarizing", strength=0.05, trajectories=128,
                               seeds=(0, 1, 2))
    print(noisy)
    print(RISK_SLOPE_HIGH, RISK_VARIANCE_HIGH)


    # ('hardware_efficient', 'strongly_entangling')
    # TrainabilityConfig(ansatz='hardware_efficient', n_qubits=4, depth=1, noise=None, strength=0.0, trajectories=32, seeds=(0, 1), dtype=complex64, device=device(type='cpu'), state_storage=None)
    # 8 ['ansatz', 'depth', 'effective_n', 'grad_kurtosis_excess', 'grad_variance', 'grad_variance_mean_marginal', 'mean_gradient', 'n_coefficients', 'n_parameters', 'n_points', 'n_qubits', 'noise', 'rse', 'samples_per_seed', 'seed', 'strength', 'trajectories']
    # hardware_efficient 4 0.0168729 medium -2.171
    # hardware_efficient 6 0.00699567 medium -2.171
    # strongly_entangling 4 0.00654054 medium -1.358
    # strongly_entangling 6 0.00377074 medium -1.358
    # cap_width : Cap hardware_efficient circuits at depth 1 to n_qubits <= 4: the gradient variance falls 2.4x from width 4 to 6 (log-log slope -2.17).
    # start_config : Start training from hardware_efficient n=4 depth=1 noise=none trajectories=32: it has the largest measured mean gradient variance (0.01687).
    # seed_precision : Treat the seed mean as resolved to +/-368.8% on average; add seeds or --samples-per-seed before comparing sub-10% differences.
    # risk_summary : 0 of 4 measured configurations are high risk; the smallest gradient variance is 0.003771 at strongly_entangling n=6 depth=1 noise=none trajectories=32.
    # (4, 1)
    # TrainabilityConfig(ansatz='hardware_efficient', n_qubits=4, depth=2, noise='depolarizing', strength=0.05, trajectories=128, seeds=(0, 1, 2), dtype=complex64, device=device(type='cpu'), state_storage=None)
    # -3.0 0.001

``pyvqnet.qnn.vqc.diagnostics`` 同时提供生成完整报告的命令行脚本 ``pyVQNet/examples/qml_alg/trainability_report.py`` (``--quick`` 使用小规模矩阵)。

.. _benchmarks:

量子机器学习批量数据训练基准测试
===========================================

 

测试1：批量数据梯度对比
------------------------------------------------------------

在量子机器学习中，梯度计算是影响变分量子线路效率的关键环节。为评估不同框架的量子梯度计算性能，本文在 Linux 系统下使用 GPU，对 VQNet 与 Pennylane 等进行了基准测试。测试在不同批量数据规模（batch size 16、32）、线路深度（layer 2、4）与量子比特数目（qubit 4、8、12、16）下分别进行，统计了各框架运行 10 次的平均耗时。线路结构为 CNOT + RX + RZ + RX 编码层。其中 Pennylane 等基于 Torch 的 GPU 后端实现，VQNet 则使用自研代码实现 GPU 运算。

.. image:: ./images/grad-benchmarks.png
   :width: 600 px
   :align: center

|



.. code-block::

    import time
    import json
    from functools import reduce
    import numpy as np
    import pennylane as qml

    import matplotlib.pyplot as plt
    def benchmark(f, *args, trials=10, sync_fn=None):
        time0 = time.time()
        r = f(*args)
        if sync_fn:
            sync_fn(r)
        time1 = time.time()
        for _ in range(trials):
            r = f(*args)
        if sync_fn:
            sync_fn(r)
        time2 = time.time()
        if trials > 0:
            time21 = (time2 - time1) / trials
        else:
            time21 = 0
        ts = (time1 - time0, time21)

        print('staging time: %.6f s' % ts[0])
        if trials > 0:
            print('running time: %.6f s' % ts[1])
        return r, ts[1]

    import torch
    import deepquantum as dq
    def grad_dq(b, n, l, trials=10):
        def get_grad_dq(params):
            if params.grad != None:
                params.grad.zero_()
            cir = dq.QubitCircuit(n)
            for j in range(l):

                for i in range(n - 1):
                    cir.cnot(i, i + 1)
                cir.rxlayer(encode=True)
                cir.rzlayer(encode=True)
                cir.rxlayer(encode=True)
            for w in range(n):
                cir.observable(basis='z',wires=w)
            cir.to("cuda:0")
            cir(data=params)
            exp = cir.expectation()
            exp.backward(torch.ones_like(exp))
            return params.grad


        return benchmark(get_grad_dq, torch.ones([b,3 * n * l], requires_grad=True,device="cuda:0"),
                         sync_fn=lambda _: torch.cuda.synchronize())


    def grad_pl_torchlayer(b, n, l,t):

        dev = qml.device("default.qubit", wires=n)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs,weights):
            params = inputs

            for j in range(l):
                for i in range(n - 1):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(n):
                    qml.RX(params[:,3 * n * j + i], i)
                for i in range(n):
                    qml.RZ(params[:,3 * n * j + i + n], i)
                for i in range(n):
                    qml.RX(params[:,3 * n * j + i + 2 * n], i)

            obs = reduce(lambda x, y: x @ y, [qml.PauliZ(i) for i in range(n)])
            y = qml.expval(obs)
            return y

        def get_grad_pl(params):
            params.grad = None
            weight_shapes = {"weights": 1}

            qlayer = qml.qnn.TorchLayer(circuit, weight_shapes = weight_shapes)
            qlayer.to("cuda:0")
            y = qlayer(params)

            y.backward(torch.ones_like(y))
            return params.grad
        return benchmark(get_grad_pl, torch.ones([b,3 * n * l],device="cuda:0", requires_grad=True),trials=t,
                         sync_fn=lambda _: torch.cuda.synchronize())

    def grad_pyvqnet_vqc(b, n, l, t):
        from pyvqnet.qnn.vqc import QMachine,cnot,rx,rz,ry,MeasureAll
        from pyvqnet.tensor import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet-ad")
        def pqctest(qm,param):
            param.zero_grad()
            qm.reset_states(param.shape[0])
            for j in range(l):
                for i in range(n - 1):
                    cnot(qm,[i, i + 1])
                for i in range(n):
                    rx(qm, i, param[:,3 * n * j + i])
                for i in range(n):
                    rz(qm, i, param[:,3 * n * j + i + n])
                for i in range(n):
                    rx(qm, i, param[:,3 * n * j + i + 2 * n])
            pauli_str =""
            for position in range(n):
                pauli_str += "Z" + str(position)+" "
            p_dict = {pauli_str:1}
            ma = MeasureAll(obs=p_dict)

            y = ma(qm)
            y.backward()

            return param.grad

        def get_grad(qm,values):
            r = pqctest(qm,values)

            return r

        input = tensor.ones([b,3 * n * l],device=pyvqnet.DEV_GPU)
        input.requires_grad = True
        qm = QMachine(n)
        qm.toGPU(pyvqnet.DEV_GPU)
        return benchmark(get_grad, qm, input, trials=t,
                         sync_fn=lambda r: r.numpy())



    N_LIST = [ 4,8,12,16 ]
    L_LIST =[2,4]
    B_LIST =[16, 32]
    def test_1():
        results ={}
        config_key = []
        for n in N_LIST:
            for l in L_LIST:
                for b in B_LIST:
                    for t in [10,]:
                        config_key.append(str(b) + '-' + str(n) + '-' + str(l))

                        dqr, ts1 = grad_pl_torchlayer(b,n, l, t)
                        results[str(b) + '-' + str(n) + '-' + str(l) + '-' + 'grad' + '-pl'] = ts1
                        print(f'pennylane 批次数：{b} 比特数：{n} 线路层数: {l}  迭代: {t}, 计算grad平均耗时 {ts1:4f}秒' )

                        dqr, ts3 = grad_dq(b, n, l, t)
                        results[str(b) + '-' + str(n) + '-' + str(l) + '-' + 'grad' + '-dq'] = ts3
                        print(f'dq 批次数：{b} 比特数：{n} 线路层数: {l}  迭代: {t}, 计算grad平均耗时 {ts3:4f}秒' )


                        result, ts2 = grad_pyvqnet_vqc(b, n, l, t)
                        results[str(b) + '-' + str(n) + '-' + str(l) + '-' + 'grad' + '-pyvqnet'] = ts2
                        print(f'pyvqnet 批次数：{b} 比特数：{n} 线路层数: {l}  迭代: {t}, 计算grad平均耗时 {ts2:4f}秒' )


        with open('gradient_results.data', 'w') as f:
            json.dump(results, f)

        with open('gradient_results.data', 'r') as f:
            results = json.load(f)

        data = results

        sub_w  = 2
        sub_l = int(len(N_LIST)/2)
        assert len(N_LIST)%2==0
        fig, axes = plt.subplots(sub_w, sub_l, figsize=(12, 10))
        ax_i=0
        for n in N_LIST:
            config_key = []
            for l in L_LIST:
                for b in B_LIST:
                    for t in [10,]:
                        config_key.append(str(b) + '-' + str(n) + '-' + str(l))
            groups = config_key
            pl_times = [data[f'{group}-grad-pl'] for group in groups]
            dq_times = [data[f'{group}-grad-dq'] for group in groups]
            pyvqnet_times = [data[f'{group}-grad-pyvqnet'] for group in groups]


            x = np.arange(len(groups))
            width = 0.15

            ax = axes[int(ax_i/sub_w), ax_i %sub_w]
            ax_i +=1
            #fig, ax = plt.subplots(figsize=(10, 6))


            rects2 = ax.bar(x , pyvqnet_times, width, label='pyvqnet')
            rects3 = ax.bar(x + width, dq_times, width, label='deepquantum')
            rects1 = ax.bar(x - width, pl_times, width, label='pennylane')

            ax.set_ylabel('cost time (sec)')
            ax.set_title(f'Gradient benchmarks of {n}-Qubit Variational Quantum Circuit.')
            ax.set_xticks(x)
            ax.set_xlabel('batchsize-qubits-layer_depth')
            ax.set_xticklabels(groups)
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            def autolabel(rects):

                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.2f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')

            autolabel(rects1)
            autolabel(rects2)
            autolabel(rects3)
        fig.tight_layout()
        plt.savefig(f"grad-benchmarks.png")

    test_1()



测试2：10量子比特 VQC 梯度对比
------------------------------------------------------------

本测试基于本源量子大模型论文中的 10 量子比特、10 层 VQC 线路，对比 VQNet、TorchQuantum（TQ）、Pennylane（PL）等框架的梯度计算性能。线路结构为：

  RY(data) → [RY(param) → CRZ(param) → RY(param) → CRZ(param)] × L

每层包含 40 个参数（4 组 × 10 qubit），参数总数 400。batch size 测试范围 1～1024。各框架 trials 次数：VQNet / TQ 为 20 次，PL 等较慢框架为 2 次（仅运行 2 次以节约时间）。

.. image:: ./images/grad_benchmarks_10q_ry_crz.png
   :width: 600 px
   :align: center

|

.. code-block:: python

    from pyvqnet.tensor import tensor
    from pyvqnet.qnn.vqc import RX, RY, RZ, crz, PauliX, PauliY, PauliZ, paulix, pauliy, pauliz, rx, ry, rz, MeasureAll, fused_multi_crz
    from pyvqnet.nn import ParameterDict, Parameter
    from pyvqnet.qnn.vqc import QModule, QMachine
    import numpy as np
    import pyvqnet
    import time

    QuantumDevice = QMachine
    class Encoder(QModule):

        def __init__(self):
            super().__init__()
            pass

        def forward(self, x, qdev):
            raise NotImplementedError

    op_name_dict = {
        "x": PauliX,
        "y": PauliY,
        "z": PauliZ,
        "rx": RX,
        "ry": RY,
        "rz": RZ
    }

    func_name_dict = {
        "x": paulix,
        "y": pauliy,
        "z": pauliz,
        "rx": rx,
        "ry": ry,
        "rz": rz
    }

    class GeneralEncoder(Encoder):
        """func_list list of dict

        """

        def __init__(self, func_list):
            super().__init__()
            self.func_list = func_list

        def forward(self, x, qdev):
            for info in self.func_list:
                if op_name_dict[info["func"]].num_params > 0:
                    params = x[:, info["input_idx"]]
                else:
                    params = None

                func_name_dict[info["func"]](qdev,
                                             wires=info["wires"],
                                             params=params)

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)


    class VQC_new(QModule):
        """VQC using fused_multi_crz - one parameter vector per layer"""

        def __init__(self, n_wires: int = 4, n_qlayers: int = 1):
            super().__init__()
            self.n_wires = n_wires
            self.n_qlayers = n_qlayers
            self.dev = QuantumDevice(self.n_wires)
            enc_cnt = list()
            for i in range(self.n_wires):
                cnt = {'input_idx': [i], 'func': 'ry', 'wires': [i]}
                enc_cnt.append(cnt)

            self.encoder = GeneralEncoder(enc_cnt)
            self._use_vqnet = True

            self.params_ry1_dct = ParameterDict()
            self.params_ry2_dct = ParameterDict()
            self.params_crx1_dct = ParameterDict()
            self.params_crx2_dct = ParameterDict()

            for k in range(self.n_qlayers):
                self.params_crx1_dct[str(k)] = Parameter([self.n_wires])
                self.params_crx2_dct[str(k)] = Parameter([self.n_wires])
                for i in range(self.n_wires):
                    self.params_ry1_dct[str(i + k * self.n_wires)] = Parameter([1])
                    self.params_ry2_dct[str(i + k * self.n_wires)] = Parameter([1])

            self.use_pq3 = False

            obs_list = []
            for i in range(self.n_wires):
                obs_list.append({f"Z{i}": 1})

            self.measure = MeasureAll(obs=obs_list)

        def forward(self, x):
            q_device = self.dev
            q_device.reset_states(x.shape[0])

            self.encoder(x, q_device)

            for k in range(self.n_qlayers):
                for i in range(self.n_wires):
                    ry(q_machine=q_device, wires=i, params=self.params_ry1_dct[str(i + k * self.n_wires)])

                obj_qubits = [(i + 1) % self.n_wires for i in range(self.n_wires - 1, -1, -1)]
                ctrls = list(range(self.n_wires - 1, -1, -1))
                fused_multi_crz(
                    q_machine=q_device,
                    params=self.params_crx1_dct[str(k)],
                    obj_qubits=obj_qubits,
                    ctrls=ctrls)

                for i in range(self.n_wires):
                    ry(q_machine=q_device, params=self.params_ry2_dct[str(i + k * self.n_wires)], wires=i)

                obj_qubits = [(i - 1) % self.n_wires for i in [self.n_wires - 1] + list(range(self.n_wires - 1))]
                ctrls = [self.n_wires - 1] + list(range(self.n_wires - 1))
                fused_multi_crz(
                    q_machine=q_device,
                    params=self.params_crx2_dct[str(k)],
                    obj_qubits=obj_qubits,
                    ctrls=ctrls)

            if self.use_pq3:
                return x
            else:
                return self.measure(q_device)


    class VQC(QModule):
        """VQC using individual crz gates - one parameter per gate"""

        def __init__(self, n_wires: int = 4, n_qlayers: int = 1):
            super().__init__()
            self.n_wires = n_wires
            self.n_qlayers = n_qlayers
            self.dev = QuantumDevice(self.n_wires)
            enc_cnt = list()
            for i in range(self.n_wires):
                cnt = {'input_idx': [i], 'func': 'ry', 'wires': [i]}
                enc_cnt.append(cnt)

            self.encoder = GeneralEncoder(enc_cnt)
            self._use_vqnet = True

            self.params_ry1_dct = ParameterDict()
            self.params_ry2_dct = ParameterDict()
            self.params_crx1_dct = ParameterDict()
            self.params_crx2_dct = ParameterDict()

            for k in range(self.n_qlayers):
                for i in range(self.n_wires):
                    self.params_crx1_dct[str(i + k * self.n_wires)] = Parameter([1])
                    self.params_crx2_dct[str(i + k * self.n_wires)] = Parameter([1])
                    self.params_ry1_dct[str(i + k * self.n_wires)] = Parameter([1])
                    self.params_ry2_dct[str(i + k * self.n_wires)] = Parameter([1])

            self.use_pq3 = False

            obs_list = []
            for i in range(self.n_wires):
                obs_list.append({f"Z{i}": 1})

            self.measure = MeasureAll(obs=obs_list)

        def forward(self, x):
            q_device = self.dev
            q_device.reset_states(x.shape[0])

            self.encoder(x, q_device)

            for k in range(self.n_qlayers):
                for i in range(self.n_wires):
                    ry(q_machine=q_device, wires=i, params=self.params_ry1_dct[str(i + k * self.n_wires)])

                for i in range(self.n_wires - 1, -1, -1):
                    crz(
                        q_machine=q_device,
                        params=self.params_crx1_dct[str(i + k * self.n_wires)],
                        wires=[i, (i + 1) % self.n_wires])

                for i in range(self.n_wires):
                    ry(q_machine=q_device, params=self.params_ry2_dct[str(i + k * self.n_wires)], wires=i)

                for i in [self.n_wires - 1] + list(range(self.n_wires - 1)):
                    crz(
                        q_machine=q_device,
                        params=self.params_crx2_dct[str(i + k * self.n_wires)],
                        wires=[i, (i - 1) % self.n_wires])

            if self.use_pq3:
                return x
            else:
                return self.measure(q_device)


    def benchmark(f, *args, trials=10, sync_fn=None):
        time0 = time.time()
        r = f(*args)
        if sync_fn:
            sync_fn(r)
        time1 = time.time()
        for _ in range(trials):
            r = f(*args)
        if sync_fn:
            sync_fn(r)
        time2 = time.time()
        if trials > 0:
            time21 = (time2 - time1) / trials
        else:
            time21 = 0
        ts = (time1 - time0, time21)
        print('staging time: %.6f s' % ts[0])
        if trials > 0:
            print('running time: %.6f s' % ts[1])
        return r, ts


    def grad_pyvqnet_vqc_new(b, n, l, trials=10):
        """Test VQC_new (fused_multi_crz)"""
        pyvqnet.utils.set_random_seed(42)
        layer = VQC_new(n, l)
        layer.toGPU(pyvqnet.DEV_GPU)

        def get_grad(values):
            r = layer(values)
            r.backward()
            return values.grad

        input = tensor.ones([b, n], device=pyvqnet.DEV_GPU)
        input.requires_grad = True
        return benchmark(get_grad, input, trials=trials,
                         sync_fn=lambda r: r.numpy())


    def grad_pyvqnet_vqc(b, n, l, trials=10):
        pyvqnet.utils.set_random_seed(42)
        layer = VQC(n, l)
        layer.toGPU(pyvqnet.DEV_GPU)

        def get_grad(values):
            r = layer(values)
            r.backward()
            return values.grad

        input = tensor.ones([b, n], device=pyvqnet.DEV_GPU)
        input.requires_grad = True
        return benchmark(get_grad, input, trials=trials,
                         sync_fn=lambda r: r.numpy())


    def grad_tq_vqc_with_params(b, n, l, trials=10):
        """TorchQuantum VQC benchmark"""
        import torchquantum as tq
        import torch
        import torch.cuda

        class VQC_TQ(tq.QuantumModule):
            """TorchQuantum VQC matching VQNet's VQC structure"""

            def __init__(self, n_wires: int = 4, n_qlayers: int = 1):
                super().__init__()
                self.n_wires = n_wires
                self.n_qlayers = n_qlayers

                enc_cnt = list()
                for i in range(self.n_wires):
                    cnt = {'input_idx': [i], 'func': 'ry', 'wires': [i]}
                    enc_cnt.append(cnt)
                self.encoder = tq.GeneralEncoder(enc_cnt)

                self.params_ry1_dct = tq.QuantumModuleDict()
                self.params_ry2_dct = tq.QuantumModuleDict()
                self.params_crx1_dct = tq.QuantumModuleDict()
                self.params_crx2_dct = tq.QuantumModuleDict()

                for k in range(self.n_qlayers):
                    for i in range(self.n_wires):
                        self.params_ry1_dct[str(i + k * self.n_wires)] = tq.RY(trainable=True)
                        self.params_crx1_dct[str(i + k * self.n_wires)] = tq.CRZ(trainable=True)
                        self.params_ry2_dct[str(i + k * self.n_wires)] = tq.RY(trainable=True)
                        self.params_crx2_dct[str(i + k * self.n_wires)] = tq.CRZ(trainable=True)

                self.measure = tq.MeasureMultipleTimes([{'wires': range(self.n_wires), 'observables': ['z'] * self.n_wires}])

                from torchquantum import QuantumDevice as TQQuantumDevice
                self.dev = TQQuantumDevice(self.n_wires)

            def forward(self, x: torch.Tensor):
                q_device = self.dev
                q_device.reset_states(x.shape[0])
                self.encoder(q_device, x)

                for k in range(self.n_qlayers):
                    for i in range(self.n_wires):
                        self.params_ry1_dct[str(i + k * self.n_wires)](q_device, wires=i)

                    for i in range(self.n_wires - 1, -1, -1):
                        self.params_crx1_dct[str(i + k * self.n_wires)](q_device, wires=[i, (i + 1) % self.n_wires])

                    for i in range(self.n_wires):
                        self.params_ry2_dct[str(i + k * self.n_wires)](q_device, wires=i)

                    for i in [self.n_wires - 1] + list(range(self.n_wires - 1)):
                        self.params_crx2_dct[str(i + k * self.n_wires)](q_device, wires=[i, (i - 1) % self.n_wires])

                return self.measure(q_device)

        torch.manual_seed(42)
        layer = VQC_TQ(n, l)
        layer.to("cuda:0")

        def get_grad(values):
            r = layer(values)
            r.backward(torch.ones_like(r))
            return values.grad

        input = torch.ones([b, n], device="cuda:0")
        input.requires_grad = True
        return benchmark(get_grad, input, trials=trials,
                         sync_fn=lambda _: torch.cuda.synchronize())

    # ──────────────────────────────────────────────
    # PennyLane benchmark
    # ──────────────────────────────────────────────

    def grad_pl_vqc(b, n, l, trials=1):
        """PennyLane VQC (default.qubit) matching VQNet VQC structure."""
        import pennylane as qml
        from functools import reduce
        import torch
        dev = qml.device("default.qubit", wires=n)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights_ry1, weights_crz1, weights_ry2, weights_crz2):
            for j in range(l):
                for i in range(n):
                    qml.RY(inputs[:, i], wires=i)
                for i in range(n):
                    qml.RY(weights_ry1[j, i], wires=i)
                for i in range(n - 1, -1, -1):
                    qml.CRZ(weights_crz1[j, i], wires=[i, (i + 1) % n])
                for i in range(n):
                    qml.RY(weights_ry2[j, i], wires=i)
                for i in [n - 1] + list(range(n - 1)):
                    qml.CRZ(weights_crz2[j, i], wires=[i, (i - 1) % n])

            obs = reduce(lambda x, y: x @ y, [qml.PauliZ(i) for i in range(n)])
            return qml.expval(obs)

        weight_shapes = {
            "weights_ry1": (l, n),
            "weights_crz1": (l, n),
            "weights_ry2": (l, n),
            "weights_crz2": (l, n),
        }

        def get_grad_pl(inputs):
            torch.manual_seed(42)
            qlayer = qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes)
            qlayer.to("cuda:0")
            y = qlayer(inputs)
            y.backward(torch.ones_like(y))
            return inputs.grad

        params = torch.ones([b, n], device="cuda:0", requires_grad=True)
        return benchmark(get_grad_pl, params, trials=trials)


    # ──────────────────────────────────────────────
    # DeepQuantum benchmark
    # ──────────────────────────────────────────────

    def grad_dq_vqc(b, n, l, trials=10):
        """DeepQuantum VQC matching VQNet VQC structure."""
        import deepquantum as dq
        import torch
        import torch.cuda
        def get_grad_dq(input_data):
            cir = dq.QubitCircuit(n, reupload=True)
            for j in range(l):
                for i in range(n):
                    cir.ry(wires=i, encode=True)
                for i in range(n):
                    cir.ry(wires=i)
                for i in range(n - 1, -1, -1):
                    cir.crz(control=i, target=(i + 1) % n)
                for i in range(n):
                    cir.ry(wires=i)
                for i in [n - 1] + list(range(n - 1)):
                    cir.crz(control=i, target=(i - 1) % n)
            for w in range(n):
                cir.observable(basis='z', wires=w)
            cir.to("cuda:0")
            cir(data=input_data)
            exp = cir.expectation()
            exp.backward(torch.ones_like(exp))
            return input_data.grad

        params = torch.ones([b, n], device="cuda:0", requires_grad=True)
        return benchmark(get_grad_dq, params, trials=trials,
                         sync_fn=lambda _: torch.cuda.synchronize())


    # ──────────────────────────────────────────────
    # MindQuantum benchmark
    # ──────────────────────────────────────────────

    def grad_mq_vqc(b, n, l, trials=1):
        """MindQuantum VQC with mqvector_gpu (cuQuantum) backend."""
        from mindquantum.core.circuit import Circuit
        from mindquantum.core.gates import RY, RZ, X
        from mindquantum.core.operators import Hamiltonian, QubitOperator
        from mindquantum.simulator import Simulator

        total_circuit = Circuit()
        for j in range(l):
            layer_enc = Circuit()
            for i in range(n):
                layer_enc += RY(f'enc_{j}_{i}').on(i)
            layer_enc.as_encoder()

            layer_ans = Circuit()
            for i in range(n):
                layer_ans += RY(f'ry1_{j}_{i}').on(i)
            for i in range(n - 1, -1, -1):
                tgt = (i + 1) % n
                ctrl = i
                p = f'crz1_{j}_{i}'
                layer_ans += RZ({p: 0.5}).on(tgt)
                layer_ans += X.on(tgt, ctrl)
                layer_ans += RZ({p: -0.5}).on(tgt)
                layer_ans += X.on(tgt, ctrl)
            for i in range(n):
                layer_ans += RY(f'ry2_{j}_{i}').on(i)
            for i in [n - 1] + list(range(n - 1)):
                tgt = (i - 1) % n
                ctrl = i
                p = f'crz2_{j}_{i}'
                layer_ans += RZ({p: 0.5}).on(tgt)
                layer_ans += X.on(tgt, ctrl)
                layer_ans += RZ({p: -0.5}).on(tgt)
                layer_ans += X.on(tgt, ctrl)
            layer_ans.as_ansatz()

            total_circuit += layer_enc + layer_ans

        obs = ' '.join(f'Z{i}' for i in range(n))
        ham = Hamiltonian(QubitOperator(obs))
        sim = Simulator('mqvector_gpu', n)
        grad_ops = sim.get_expectation_with_grad(ham, total_circuit)
        n_ansatz_params = 4 * n * l
        ansatz_data = np.ones(n_ansatz_params, dtype=np.float32)

        def get_grad_mq(input_data):
            encoder_data = np.tile(input_data, (1, l)).astype(np.float32)
            _, g_enc, _ = grad_ops(encoder_data, ansatz_data)
            return g_enc

        def sync_mq(g_enc):
            g_enc_real = np.asarray(g_enc.real, dtype=np.float32)
            g = np.zeros((b, n), dtype=np.float32)
            for j in range(l):
                g += g_enc_real[:, 0, j * n : (j + 1) * n]
            return g

        input_data = np.ones((b, n), dtype=np.float32)
        return benchmark(get_grad_mq, input_data, trials=trials,
                         sync_fn=sync_mq)


    # ──────────────────────────────────────────────
    # Plotting
    # ──────────────────────────────────────────────

    def _parse_trials(results, fw):
        """Extract trials count from results keys like '*-grad-{fw}-t{N}'."""
        for k in results:
            if f'-grad-{fw}-t' in k:
                return k.split('-t')[-1]
        return '?'


    def plot_results(results, output_path="grad_benchmarks_10q_ry_crz.png"):
        """
        Line chart: x = batch size, y = running time (log scale).
        Keys format: "{batch}-{n}-{l}-grad-{framework}-t{trials}".
        """
        import matplotlib.pyplot as plt

        frameworks = ['pyvqnet', 'new', 'torchquantum', 'pl', 'dq', 'mq']
        labels_base = {
            'pyvqnet': 'pyVQNet',
            'new': 'pyVQNet (fused CRZ)',
            'torchquantum': 'TorchQuantum',
            'pl': 'PennyLane',
            'dq': 'DeepQuantum',
            'mq': 'MindQuantum',
        }
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        batch_sizes = sorted({int(k.split('-')[0]) for k in results})
        n_qubits = sorted({k.split('-')[1] for k in results})
        n_layers = sorted({k.split('-')[2] for k in results})
        n_q = int(n_qubits[0]) if n_qubits else 10
        n_l = int(n_layers[0]) if n_layers else 10

        fig, ax = plt.subplots(figsize=(10, 6))

        for fw, color in zip(frameworks, colors):
            trials_str = _parse_trials(results, fw)
            label = f"{labels_base[fw]} (t={trials_str})"
            times = []
            for bs in batch_sizes:
                candidates = [k for k in results if k.startswith(f'{bs}-{n_q}-{n_l}-grad-{fw}')]
                if candidates:
                    times.append(results[candidates[0]][1])
                else:
                    times.append(None)
            valid = [(bs, t) for bs, t in zip(batch_sizes, times) if t is not None]
            if valid:
                xs, ys = zip(*valid)
                ax.plot(xs, ys, marker='o', label=label, color=color, linewidth=2)

        ax.set_xlabel('Batch Size', fontsize=13)
        ax.set_ylabel('Running Time (s) — log scale', fontsize=13)
        ax.set_title(f'VQC Gradient Benchmark (n_qubits={n_q}, n_layers={n_l})', fontsize=14)
        ax.set_xticks(batch_sizes)
        ax.set_yscale('log')
        ax.legend(fontsize=11, loc='upper left', bbox_to_anchor=(0.02, 0.98))
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)

    def test_3():
        """
        Run all benchmarks across multiple configs.
        Fast frameworks (pyvqnet, TQ, DQ): trials=100
        Slow frameworks (PL, MQ):         trials=2
        JSON keys include trials count so results are distinguishable.
        """
        import os
        import json
        json_path = "compare_grad_calc_results.json"
        if os.path.exists(json_path):
            print(f"{json_path} already exists, loading and plotting directly...")
            with open(json_path) as f:
                results = json.load(f)
            print()
            print("=== Loaded Results ===")
            for key, ts in results.items():
                print(f"{key}: staging={ts[0]:.4f}s, running={ts[1]:.4f}s")
            plot_results(results)
            return

        results = {}
        n_list = [10,]
        l_list = [10, ]
        b_list = [1024,512,256,128,64]
        t_fast, t_slow = 20, 2

        for n in n_list:
            for l in l_list:
                for b in b_list:
                    print(str(b) + '-' + str(n) + '-' + str(l) + '-' + 'grad')
                    print("grad_pyvqnet_vqc")
                    pyvqnet.utils.set_random_seed(42)
                    result2, ts = grad_pyvqnet_vqc(b, n, l, trials=t_fast)
                    results[str(b) + '-' + str(n) + '-' + str(l) + '-' + 'grad' + f'-pyvqnet-t{t_fast}'] = ts

                    print(str(b) + '-' + str(n) + '-' + str(l) + '-' + 'grad')
                    print("grad_pyvqnet_vqc_new")
                    pyvqnet.utils.set_random_seed(42)
                    result1, ts = grad_pyvqnet_vqc_new(b, n, l, trials=t_fast)
                    results[str(b) + '-' + str(n) + '-' + str(l) + '-' + 'grad' + f'-new-t{t_fast}'] = ts

                    print(str(b) + '-' + str(n) + '-' + str(l) + '-' + 'grad')
                    print("grad_torchquantum_vqc")
                    result_tq, ts = grad_tq_vqc_with_params(b, n, l, trials=t_fast)
                    results[str(b) + '-' + str(n) + '-' + str(l) + '-' + 'grad' + f'-torchquantum-t{t_fast}'] = ts

                    print(str(b) + '-' + str(n) + '-' + str(l) + '-' + 'grad')
                    print("grad_pennylane_vqc")
                    _, ts = grad_pl_vqc(b, n, l, trials=t_slow)
                    results[f'{b}-{n}-{l}-grad-pl-t{t_slow}'] = ts

                    print(str(b) + '-' + str(n) + '-' + str(l) + '-' + 'grad')
                    print("grad_deepquantum_vqc")
                    _, ts = grad_dq_vqc(b, n, l, trials=t_fast)
                    results[f'{b}-{n}-{l}-grad-dq-t{t_fast}'] = ts

                    print(str(b) + '-' + str(n) + '-' + str(l) + '-' + 'grad')
                    print("grad_mindquantum_vqc")
                    _, ts = grad_mq_vqc(b, n, l, trials=t_slow)
                    results[f'{b}-{n}-{l}-grad-mq-t{t_slow}'] = ts

        print("\n=== All Results ===")
        for key, ts in results.items():
            print(f"{key}: staging={ts[0]:.4f}s, running={ts[1]:.4f}s")

        with open('compare_grad_calc_results.json', 'w') as f:
            json.dump(results, f)
        print("Results saved to compare_grad_calc_results.json")

        # Also generate plot
        plot_results(results)


    test_3()

+------------------+-----------+
| 项目             | 规格      |
+==================+===========+
| CPU              | i9-9900K  |
+------------------+-----------+
| 显卡             | GTX3090   |
+------------------+-----------+
| CUDA             | 12.6      |
+------------------+-----------+
| 内存             | 64GB      |
+------------------+-----------+
| deepquantum      | 4.5.0     |
+------------------+-----------+
| mindquantum      | 0.12.0    |
+------------------+-----------+
| pennylane        | 0.42.3    |
+------------------+-----------+
| pyvqnet          | 2.18.0    |
+------------------+-----------+
| torch            | 2.12.0    |
+------------------+-----------+
| torchquantum     | 0.2.0     |
+------------------+-----------+

张量网络后端变分量子线路模块
============================================

.. note::

    本模块基于 ``jax`` 实现自动微分与 GPU 加速。默认安装 ``pyvqnet`` 不包含该依赖，请使用 ``pip install jax``（CPU）或 ``pip install jax[cuda12]``（GPU，需 CUDA 12.6）安装。此外还需额外安装 ``tensornetwork`` ： ``pip install tensornetwork`` 。

张量网络（Tensor Network）通过将复杂的张量分解为多个低维张量的网络，显著降低了计算复杂度。

矩阵乘积态（Matrix Product State, MPS）是张量网络的一种特殊形式，MPS 将量子态表示为一系列矩阵的乘积，从而有效减少参数数量，降低了计算复杂度。

下面接口则是基于 ``pyvqnet`` 后端（默认后端），对张量网络构建量子线路的功能支持，包括对构建量子线路基类、量子逻辑门、量子线路以及测量方法，并通过自动微分模拟代替参数移位法计算参数梯度。

以MPS方式构建量子线路弥补对大比特量子线路构建支持。

.. warning::

        通过 ``TNQMachine`` 中 ``use_mps`` 参数开启MPS构建量子线路功能， 支持大比特(100以及以上)量子线路实现。

.. warning::
        
        批量化与经典模块下使用方式不同，基于vmap的方式，数据以及参数构建线路需降一维输入，即对应态矢模拟时代码 ``x[:,i]`` 需要改为 ``x[i]``, 具体可查看下方接口中样例, 批次化执行必须同时基于 ``TNQMachine`` 和 ``TNQModule`` 并使用 ``TNQMachine`` 的 ``reset_states`` 显式指定批次大小。

基类
--------------------------------------------------

TNQModule
^^^^^^^^^^^^^^^^^^^^^^^^

基于张量网络编写变分量子线路模型需要继承于 ``TNQModule``

.. py:class:: pyvqnet.qnn.vqc.tn.native.TNQModule(use_jit=True,vectorized_argnums: Union[int, Sequence[int]]=0, name="", mesh=None, cotengra_options=None, tree_path=None)

    在 `pyvqnet` 后端（默认后端）下,定义张量网络下量子变分线路模型 `Module` 应该继承的基类。
    该类用于使用张量网络来模块来用语执行量子线路。

    .. warning::

        ``TNQModule`` 仅支持量子类型的子模块（``TNQMachine``、量子门、ansatz 模板、
        测量模块与编码模块）。经典神经网络层（例如 ``Linear``）不可作为 ``TNQModule``
        的子模块：JIT 编译的对象是纯量子线路，其中的经典计算将被忽略，或在编译时
        被固定为常量。构建经典-量子混合模型时，应将经典层定义在外层普通
        ``pyvqnet.nn.Module`` 中，并将 ``TNQModule`` 作为其子模块；如需不经 JIT
        编译执行，可直接使用 ``TNQMachine``。正确的组织方式参见各 ansatz 接口示例
        中 ``QPart``/``QM`` 的双层结构。

    :param use_jit: 开启即时编译功能, 默认为True。
    :param vectorized_argnums: 要被向量化的参数,这些参数应该在同一维共享相同的批次形状,默认值为0
    :param name: 模块名。
    :param mesh: 用于分布式收缩的 ``jax.sharding.Mesh``。为 ``None`` 时使用单设备 JIT; 指定 mesh 时通过 SPMD 实现模型并行。
    :param cotengra_options: 可选的字典,在指定 ``mesh`` 时传递给 ``cotengra.ReusableHyperOptimizer``。
    :param tree_path: 可选的预计算收缩路径文件(``.pkl``),指定后将跳过运行时的 cotengra 搜索。

    .. note::

        开启 ``use_jit`` 后，模型会使用 ``jax`` 的 ``jit`` 进行即时编译，首次运行将会进行编译，会耗时较长。

    .. note::

        该类以及其派生类仅适用于 ``pyvqnet.backends.set_backend("pyvqnet")`` 。

    Example::

        import pyvqnet
        from pyvqnet.nn import Parameter
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import TNQModule
        from pyvqnet.qnn.vqc.tn.native import TNQMachine, RX, RY, CNOT, PauliX, PauliZ,qmeasure,qcircuit,VQC_RotCircuit
        class QModel(TNQModule):
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = TNQMachine(num_wires, dtype=dtype)

                self.w = Parameter((2,4,3),initializer=pyvqnet.utils.initializer.quantum_uniform)
                self.cnot = CNOT(wires=[0, 1])
                self.batch_size = batch_size
            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(batchsize=self.batch_size)

                def get_cnot(nqubits,qm):
                    for i in range(len(nqubits) - 1):
                        CNOT(wires = [nqubits[i], nqubits[i + 1]])(q_machine = qm)
                    CNOT(wires = [nqubits[len(nqubits) - 1], nqubits[0]])(q_machine = qm)


                def build_circuit(weights, xx, nqubits,qm):
                    def Rot(weights_j, nqubits,qm):#pylint:disable=invalid-name
                        VQC_RotCircuit(qm,nqubits,weights_j)

                    def basisstate(qm,xx, nqubits):
                        for i in nqubits:
                            qcircuit.rz(q_machine=qm, wires=i, params=xx[i])
                            qcircuit.ry(q_machine=qm, wires=i, params=xx[i])
                            qcircuit.rz(q_machine=qm, wires=i, params=xx[i])

                    basisstate(qm,xx,nqubits)

                    for i in range(weights.shape[0]):

                        weights_i = weights[i, :, :]
                        for j in range(len(nqubits)):
                            weights_j = weights_i[j]
                            Rot(weights_j, nqubits[j],qm)
                        get_cnot(nqubits,qm)

                build_circuit(self.w, x,range(4),self.qm)

                y= qmeasure.MeasureAll(obs={'Z0': 1})(self.qm)
                return y


        x= pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        y.backward(pyvqnet.tensor.ones_like(y))

TNQMachine
^^^^^^^^^^^^^^^^^^^^^^^^

基于张量网络编写变分量子线路设备需要 ``TNQMachine`` 进行初始化。 

.. py:class:: pyvqnet.qnn.vqc.tn.native.TNQMachine(num_wires, dtype=pyvqnet.kcomplex64,use_mps=False)

    变分量子计算的模拟器类,包含states属性为量子线路的statevectors。

    .. warning::
        
        在每次运行一个完整的量子线路之前,必须使用 `pyvqnet.qnn.vqc.tn.native.TNQMachine.reset_states(batchsize)` 将模拟器里面初态重新初始化,并且广播为
        (batchsize,*) 维度从而适应批量数据训练。

    .. warning::
        
        在张量网络的量子线路中，默认会开启 ``vmap`` 功能，在线路上的逻辑门参数上均为舍弃了批次维度，
        使用时，调用参数若维度为 [batch_size, \*], 在使用时舍弃第一个batch_size维度, 直接使用后面维度, 如对输入数据x[:,1] -> x[1], 对可训练参数也一致，可参考下列样例中xx, weights用法.


    :param num_wires: 量子比特数。
    :param dtype: 计算数据的数据类型。默认值是pyvqnet.kcomplex64,对应的参数精度为pyvqnet.kfloat32。
    :param use_mps: 是否基于mpscircuit进行模拟, 用于模拟大比特量子线路执行。

    :return: 输出一个TNQMachine对象。

    Example::
        
        import pyvqnet
        from pyvqnet.nn import Parameter
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import TNQModule
        from pyvqnet.qnn.vqc.tn.native import TNQMachine, RX, RY, CNOT, PauliX, PauliZ,qmeasure,qcircuit,VQC_RotCircuit
        class QModel(TNQModule):
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = TNQMachine(num_wires, dtype=dtype)

                self.w = Parameter((2,4,3),initializer=pyvqnet.utils.initializer.quantum_uniform)
                self.cnot = CNOT(wires=[0, 1])
                self.batch_size = batch_size
            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(batchsize=self.batch_size)

                def get_cnot(nqubits,qm):
                    for i in range(len(nqubits) - 1):
                        CNOT(wires = [nqubits[i], nqubits[i + 1]])(q_machine = qm)
                    CNOT(wires = [nqubits[len(nqubits) - 1], nqubits[0]])(q_machine = qm)


                def build_circuit(weights, xx, nqubits,qm):
                    def Rot(weights_j, nqubits,qm):#pylint:disable=invalid-name
                        VQC_RotCircuit(qm,nqubits,weights_j)

                    def basisstate(qm,xx, nqubits):
                        for i in nqubits:
                            qcircuit.rz(q_machine=qm, wires=i, params=xx[i])
                            qcircuit.ry(q_machine=qm, wires=i, params=xx[i])
                            qcircuit.rz(q_machine=qm, wires=i, params=xx[i])

                    basisstate(qm,xx,nqubits)

                    for i in range(weights.shape[0]):

                        weights_i = weights[i, :, :]
                        for j in range(len(nqubits)):
                            weights_j = weights_i[j]
                            Rot(weights_j, nqubits[j],qm)
                        get_cnot(nqubits,qm)

                build_circuit(self.w, x,range(4),self.qm)

                y= qmeasure.MeasureAll(obs={'Z0': 1})(self.qm)
                return y


        x= pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        y.backward(pyvqnet.tensor.ones_like(y))

    .. py:method:: get_states()

        获得张量网络中的states。

变分量子逻辑门模块
--------------------------------------------------


以下 ``pyvqnet.qnn.vqc`` 中的函数接口直接支持 ``pyvqnet`` 后端的 ``QTensor`` 进行计算，通过 ``pyvqnet.qnn.vqc.tn`` 下调用使用。

.. csv-table:: 已支持pyvqnet.qnn.vqc.sv.native接口列表
   :file: ./images/same_apis_from_tn.csv

以下量子线路模块继承于 ``pyvqnet.qnn.vqc.tn.native.TNQModule``,其中计算使用 ``QTensor`` 进行计算。


.. warning::

    该类以及其派生类仅适用于 ``pyvqnet.backends.set_backend("pyvqnet")`` 。

    这些类如果有非参数成员变量 ``_buffers`` ,则其中的数据为 ``QTensor`` 类型。
    这些类如果有参数成员变量 ``_parmeters`` ,则其中的数据为 ``pyvqnet.nn.Parameter`` 类型。

I
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.I(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个I逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 I 逻辑门实例

    Example::


        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import I,TNQMachine
        device = TNQMachine(4)
        layer = I(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.get_states())

Hadamard
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.Hadamard(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个Hadamard逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 Hadamard 逻辑门实例

    Example::


        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import Hadamard,TNQMachine
        device = TNQMachine(4)
        layer = Hadamard(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.get_states())

T
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.T(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个T逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 T 逻辑门实例

    Example::


        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import T,TNQMachine
        device = TNQMachine(4)
        layer = T(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.get_states())

S
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.S(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个S逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 S 逻辑门实例。

    Example::


        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import S,TNQMachine
        device = TNQMachine(4)
        layer = S(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.get_states())

PauliX
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.PauliX(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个PauliX逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。


    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 PauliX 逻辑门实例。

    Example::


        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import PauliX,TNQMachine
        device = TNQMachine(4)
        layer = PauliX(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.get_states())

PauliY
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.PauliY(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个PauliY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。


    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 PauliY 逻辑门实例。

    Example::


        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import PauliY,TNQMachine
        device = TNQMachine(4)
        layer = PauliY(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.get_states())

PauliZ
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.PauliZ(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个PauliZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。


    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 PauliZ 逻辑门实例。

    Example::


        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import PauliZ,TNQMachine
        device = TNQMachine(4)
        layer = PauliZ(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.get_states())

RX
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.RX(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RX逻辑门类 。


    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RX 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import RX,TNQMachine
        device = TNQMachine(4)
        layer = RX(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

RY
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.RY(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RY 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import RY,TNQMachine
        device = TNQMachine(4)
        layer = RY(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

RZ
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.RZ(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RZ 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import RZ,TNQMachine
        device = TNQMachine(4)
        layer = RZ(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

CRX
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.CRX(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CRX逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CRX 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import CRX,TNQMachine
        device = TNQMachine(4)
        layer = CRX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

CRY
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.CRY(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CRY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CRY 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import CRY,TNQMachine
        device = TNQMachine(4)
        layer = CRY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

CRZ
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.native.CRZ(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CRZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CRZ 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import CRZ,TNQMachine
        device = TNQMachine(4)
        layer = CRZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

U1
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.U1(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个U1逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 U1 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import U1,TNQMachine
        device = TNQMachine(4)
        layer = U1(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

U2
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.native.U2(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个U2逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 U2 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import U2,TNQMachine
        device = TNQMachine(4)
        layer = U2(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

U3
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.native.U3(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个U3逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 U3 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import U3,TNQMachine
        device = TNQMachine(4)
        layer = U3(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

CNOT
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.CNOT(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CNOT逻辑门类,也可称为CX。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CNOT 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import CNOT,TNQMachine
        device = TNQMachine(4)
        layer = CNOT(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

CY
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.CY(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CY 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import CY,TNQMachine
        device = TNQMachine(4)
        layer = CY(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

CZ
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.CZ(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CZ 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import CZ,TNQMachine
        device = TNQMachine(4)
        layer = CZ(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

CR
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.CR(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CR逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CR 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import CR,TNQMachine
        device = TNQMachine(4)
        layer = CR(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

SWAP
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.native.SWAP(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个SWAP逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 SWAP 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import SWAP,TNQMachine
        device = TNQMachine(4)
        layer = SWAP(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

CSWAP
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.CSWAP(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个SWAP逻辑门类 。

    .. math:: CSWAP = \begin{bmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
        \end{bmatrix}.

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CSWAP 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import CSWAP,TNQMachine
        device = TNQMachine(4)
        layer = CSWAP(wires=[0,1,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

RXX
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.native.RXX(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RXX逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RXX 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import RXX,TNQMachine
        device = TNQMachine(4)
        layer = RXX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

RYY
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.RYY(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RYY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RYY 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import RYY,TNQMachine
        device = TNQMachine(4)
        layer = RYY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

RZZ
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.RZZ(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RZZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个RZZ 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import RZZ,TNQMachine
        device = TNQMachine(4)
        layer = RZZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

RZX
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.RZX(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RZX逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RZX 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import RZX,TNQMachine
        device = TNQMachine(4)
        layer = RZX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

Toffoli
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.native.Toffoli(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个Toffoli逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 Toffoli 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import Toffoli,TNQMachine
        device = TNQMachine(4)
        layer = Toffoli(  wires=[0,2,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

IsingXX
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.native.IsingXX(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个IsingXX逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 IsingXX 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import IsingXX,TNQMachine
        device = TNQMachine(4)
        layer = IsingXX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

IsingYY
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.native.IsingYY(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个IsingYY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 IsingYY 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import IsingYY,TNQMachine
        device = TNQMachine(4)
        layer = IsingYY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

IsingZZ
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.IsingZZ(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个IsingZZ逻辑门类 。


    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 IsingZZ 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import IsingZZ,TNQMachine
        device = TNQMachine(4)
        layer = IsingZZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

IsingXY
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.native.IsingXY(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个IsingXY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 IsingXY 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import IsingXY,TNQMachine
        device = TNQMachine(4)
        layer = IsingXY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

PhaseShift
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.native.PhaseShift(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个PhaseShift逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 PhaseShift 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import PhaseShift,TNQMachine
        device = TNQMachine(4)
        layer = PhaseShift(has_params= True, trainable= True, wires=1)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

MultiRZ
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.MultiRZ(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个MultiRZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 MultiRZ 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import MultiRZ,TNQMachine
        device = TNQMachine(4)
        layer = MultiRZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())


ControlledPhaseShift
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.native.ControlledPhaseShift(trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个ControlledPhaseShift逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 ControlledPhaseShift 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import ControlledPhaseShift,TNQMachine
        device = TNQMachine(4)
        layer = ControlledPhaseShift(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

常见测量接口
--------------------------------------

VQC_Purity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.native.VQC_Purity(qubits_idx, num_wires, qmachine)

    从态矢中计算特定量子比特 ``qubits_idx`` 上的纯度。

    .. math::
        \gamma = \text{Tr}(\rho^2)

    式中 :math:`\rho` 为密度矩阵。标准化量子态的纯度满足 :math:`\frac{1}{d} \leq \gamma \leq 1` ,
    其中 :math:`d` 是希尔伯特空间的维数。
    纯态的纯度是1。

    :param qubits_idx: 要计算纯度的量子比特位索引
    :param num_wires: 量子比特数
    :param qmachine: TNQMachine 或 TNQModule 实例。

    :return: 对应比特位置上的纯度。

    .. note::
        
        批量化必须搭配TNQModule使用。

    Example::

        import pyvqnet
        from pyvqnet.qnn.vqc.tn.native import TNQMachine, qcircuit, TNQModule,VQC_Purity
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.tensor import QTensor

        x = QTensor([[0.7, 0.4], [1.7, 2.4]], requires_grad=True).toGPU()
        x.requires_grad = True
        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name)
                self.device = TNQMachine(3)
                
            def forward(self, x):
                self.device.reset_states(2)
                qcircuit.rx(q_machine=self.device, wires=0, params=x[0])
                qcircuit.ry(q_machine=self.device, wires=1, params=x[1])
                qcircuit.ry(q_machine=self.device, wires=2, params=x[1])
                qcircuit.cnot(q_machine=self.device, wires=[0, 1])
                qcircuit.cnot(q_machine=self.device, wires=[2, 1])
                return VQC_Purity([0, 1], 3, self.device)

        model = QM().toGPU()
        y_tn = model(x)
        
        y_tn.backward(pyvqnet.tensor.ones_like(y_tn))
        print(y_tn)

VQC_DensityMatrixFromQstate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.native.VQC_DensityMatrixFromQstate(indices, num_wires, qmachine)

    计算量子态在一组特定量子比特上的密度矩阵。

    :param indices: 所考虑子系统中的量子比特索引列表。
    :param num_wires: 量子比特数。
    :param qmachine: TNQMachine 或 TNQModule 实例。
    :return: 大小为“(b, 2**len(indices), 2**len(indices))”的密度矩阵,其中b为 q_machine.reset_states(b)的批处理数量b。

    Example::

        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn.native import TNQMachine, qcircuit, VQC_DensityMatrixFromQstate,TNQModule
        pyvqnet.backends.set_backend("pyvqnet")
        x = QTensor([[0.7,0.4],[1.7,2.4]], requires_grad=True).toGPU()
        x.requires_grad = True
        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name=name, use_jit=True)
                self.device = TNQMachine(3)
                
            def forward(self, x):
                self.device.reset_states(2)
                qcircuit.rx(q_machine=self.device, wires=0, params=x[0])
                qcircuit.ry(q_machine=self.device, wires=1, params=x[1])
                qcircuit.ry(q_machine=self.device, wires=2, params=x[1])
                qcircuit.cnot(q_machine=self.device, wires=[0, 1])
                qcircuit.cnot(q_machine=self.device, wires=[2, 1])
                return VQC_DensityMatrixFromQstate([0,1], 3, self.device)
            
        model = QM().toGPU()
        y = model(x)
 
        y.backward(pyvqnet.tensor.ones_like(y))
        print(y)

        # [[[0.8155131+0.j        0.1718155+0.j        0.       +0.0627175j
        #   0.       +0.2976855j]
        #  [0.1718155+0.j        0.0669081+0.j        0.       +0.0244234j
        #   0.       +0.0627175j]
        #  [0.       -0.0627175j 0.       -0.0244234j 0.0089152+0.j
        #   0.0228937+0.j       ]
        #  [0.       -0.2976855j 0.       -0.0627175j 0.0228937+0.j
        #   0.1086637+0.j       ]]
        # 
        # [[0.3362115+0.j        0.1471083+0.j        0.       +0.1674582j
        #   0.       +0.3827205j]
        #  [0.1471083+0.j        0.0993662+0.j        0.       +0.1131119j
        #   0.       +0.1674582j]
        #  [0.       -0.1674582j 0.       -0.1131119j 0.1287589+0.j
        #   0.1906232+0.j       ]
        #  [0.       -0.3827205j 0.       -0.1674582j 0.1906232+0.j
        #   0.4356633+0.j       ]]]   



Probability
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.native.Probability(wires, name="")

    计算量子线路在特定比特上概率测量结果。

    .. warning::
        
        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param wires: 测量比特的索引,列表、元组或者整数。
    :param name: 模块的名字,默认:""。
    :return: 测量结果,QTensor。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import Probability, qcircuit, TNQMachine, Hadamard, CNOT
        qm = TNQMachine(2)
        qm.reset_states(2)
        Hadamard(wires=0)(q_machine=qm)
        CNOT(wires=[0, 1])(q_machine=qm)
        ma = Probability(wires=1)
        y = ma(q_machine=qm)
        print(y)


MeasureAll
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.MeasureAll(obs, name="")

    计算量子线路的测量结果,支持输入obs为多个或单个泡利算子或哈密顿量。
    例如:

    {\'wires\': [0,  1], \'observables\': [\'x\', \'i\'],\'coefficient\':[0.23,-3.5]}
    或:
    {\'X0\': 0.23}
    或:
    [{\'wires\': [0, 2, 3],\'observables\': [\'X\', \'Y\', \'Z\'],\'coefficient\': [1, 0.5, 0.4]}, {\'wires\': [0, 1, 2],\'observables\': [\'X\', \'Y\', \'Z\'],\'coefficient\': [1, 0.5, 0.4]}]

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。


    :param obs: observable。
    :param name: 模块的名字,默认:""。
    :return: 一个 MeasureAll 测量方法实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import MeasureAll, qcircuit, TNQMachine, Hadamard, CNOT
        qm = TNQMachine(2)
        qm.reset_states(2)
        Hadamard(wires=0)(q_machine=qm)
        CNOT(wires=[0, 1])(q_machine=qm)
        obs_list = [{"Z0 Z1": 2}, {"X1 X0": 2}]
        ma = MeasureAll(obs=obs_list)
        y = ma(q_machine=qm)
        print(y)



Samples
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.Samples(wires=None, obs=None, shots = 1,name="")

    获取特定线路上的带有 shot 的样本结果

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。


    :param wires: 样本量子比特索引。默认值: None,根据运行时使用模拟器的所有比特。
    :param obs: 该值只能设为None。
    :param shots: 样本重复次数,默认值: 1。
    :param name: 此模块的名称,默认值: “”。
    :return: 一个 Samples 测量方法实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import Samples, TNQMachine, Hadamard, CNOT
        qm = TNQMachine(3)
        qm.reset_states(2)
        Hadamard(wires=0)(q_machine=qm)
        CNOT(wires=[0, 1])(q_machine=qm)
        ma = Samples(wires=[0, 1, 2], shots=3)
        y = ma(q_machine=qm)
        print(y)



QInfoMeasurement
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.QInfoMeasurement(quantity=None, wires=None, obs=None, name="")

    把一个量子信息量作为张量网络线路的测量模块。张量网络后端支持三个量: ``quantity="expectation"`` 计算厄密算符的期望, ``quantity="purity"`` 计算一组量子比特上的纯度, ``quantity="variance"`` 计算厄密算符的方差。 ``quantity="entropy"``、``"meyer_wallach"`` 与 ``"mutual_info"`` 在张量网络后端上抛出 ``NotImplementedError``。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    :param quantity: 测量的量, ``"expectation"``、``"purity"`` 或 ``"variance"``。 ``"expectation"`` 在 ``obs`` 是含 ``matrix`` 或 ``observables`` 键的字典时可以省略。
    :param wires: 测量作用的量子比特索引,单个索引或索引列表。 ``"purity"`` 必须给出; ``"expectation"`` 与 ``"variance"`` 的矩阵作用在整个 ``TNQMachine`` 上时可以省略。
    :param obs: 厄密矩阵。 ``"expectation"`` 可以是 ``{'wires': (1, 0), 'observables': 矩阵}`` 形式的字典,也可以直接给出矩阵; ``"variance"`` 直接给出矩阵。
    :param name: 模块的名字,默认:""。
    :return: 一个 QInfoMeasurement 测量方法实例。测量结果的形状为 [b],其中 b 为 ``TNQMachine.reset_states(b)`` 的批处理大小。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        import numpy as np
        from pyvqnet.qnn.vqc.tn.native import (
            TNQModule, TNQMachine, QInfoMeasurement, vqc_rotcircuit, RY, CNOT, PauliX, qcircuit,
        )
        from pyvqnet.qnn.vqc import rx, ry
        from pyvqnet.tensor import QTensor, tensor
        from pyvqnet.nn import Parameter

        bsz = 3
        H = np.array([[8, 4, 0, -6], [4, 0, 4, 0], [0, 4, 8, 0], [-6, 0, 0, 0]])


        class QModel(TNQModule):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()
                rot_param = Parameter((3,))
                rot_param.copy_value_from(tensor.QTensor([-0.5, 1, 2.3]))
                self.rot_param = rot_param
                self.qm = TNQMachine(num_wires, dtype=dtype)
                self.ry_layer2 = RY(trainable=True, wires=0, init_params=tensor.QTensor([-0.5]))
                self.xlayer = PauliX(wires=0)
                self.cnot = CNOT(wires=[0, 1])
                self.measure = QInfoMeasurement("expectation", obs={'wires': (1, 0), 'observables': tensor.to_tensor(H)})

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])
                rx(q_machine=self.qm, wires=0, params=x[1])
                ry(q_machine=self.qm, wires=1, params=x[0])
                self.xlayer(q_machine=self.qm)
                vqc_rotcircuit(self.qm, 1, self.rot_param)
                self.ry_layer2(q_machine=self.qm)
                self.cnot(q_machine=self.qm)
                return self.measure(q_machine=self.qm)


        input_x = tensor.arange(1, bsz * 2 + 1, dtype=pyvqnet.kfloat32).reshape([bsz, 2])
        input_x.requires_grad = True

        qunatum_model = QModel(num_wires=2, dtype=pyvqnet.kcomplex64)

        batch_y = qunatum_model(input_x)
        batch_y.backward(pyvqnet.tensor.ones_like(batch_y))
        print(batch_y)
        print(input_x.grad)
        print(qunatum_model.ry_layer2.params.grad)
        print(qunatum_model.rot_param.grad)

        Z = np.array([[1.0, 0.0], [0.0, -1.0]])
        qm = TNQMachine(2)
        qm.reset_states(1)
        qcircuit.ry(q_machine=qm, wires=1, params=QTensor([[0.4]]))
        qcircuit.cnot(q_machine=qm, wires=[1, 0])
        print(QInfoMeasurement("purity", wires=[0])(q_machine=qm))
        print(QInfoMeasurement("variance", obs=Z, wires=[0])(q_machine=qm))
        try:
            QInfoMeasurement("entropy", wires=[0])(q_machine=qm)
        except NotImplementedError as e:
            print(type(e).__name__)

        # [5.379823 ,7.1294165,0.7028296]
        # <QTensor [3] DEV_CPU float32>
        # [[-1.3423377, 5.311905 ],
        #  [-2.0510893,-2.0441792],
        #  [-0.9808328, 0.9307353]]
        # <QTensor [3, 2] DEV_CPU float32>
        # [[-2.3934531]]
        # <QTensor [1, 1] DEV_CPU float32>
        # [-0.4425026, 3.3497314, 3.6554477]
        # <QTensor [3] DEV_CPU float32>
        # [0.9241767]
        # <QTensor [1] DEV_CPU float32>
        # [0.1516466]
        # <QTensor [1] DEV_CPU float32>
        # NotImplementedError
 


常见量子线路模版
--------------------------------------------------

VQC_HardwareEfficientAnsatz
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.VQC_HardwareEfficientAnsatz(n_qubits,single_rot_gate_list,entangle_gate="CNOT",entangle_rules='linear',depth=1,initial = None,dtype=None)

    论文介绍的Hardware Efficient Ansatz的实现: `Hardware-efficient Variational Quantum Eigensolver for Small Molecules <https://arxiv.org/pdf/1704.05018.pdf>`__ 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。


    :param n_qubits: 量子比特数。
    :param single_rot_gate_list: 单个量子比特旋转门列表由一个或多个作用于每个量子比特的旋转门构成。目前支持 Rx、Ry、Rz。
    :param entangle_gate: 非参数化纠缠门。支持 CNOT、CZ。默认值: CNOT。
    :param entangle_rules: 纠缠门在电路中的使用方式。'linear' 表示纠缠门将作用于每个相邻的量子比特。'all' 表示纠缠门将作用于任意两个量子比特。默认值: linear。
    :param depth: 假设的深度,默认值: 1。
    :param initial: 使用initial 初始化所有其中参数逻辑门的参数,默认值: None,此模块将随机初始化参数。
    :param dtype: 参数的数据类型,默认值: None,使用float32。
    :return: 一个 VQC_HardwareEfficientAnsatz 实例。

    Example::

        from pyvqnet.nn import Linear, Module
        from pyvqnet.qnn.vqc.tn.native.qcircuit import VQC_HardwareEfficientAnsatz,RZZ,RZ
        from pyvqnet.qnn.vqc.tn.native import Probability,TNQMachine, TNQModule
        from pyvqnet import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        pyvqnet.utils.set_random_seed(25)

        class QPart(TNQModule):
            def __init__(self, batch_size, name=""):
                super().__init__(name)
                self.batch_size = batch_size
                self.ansatz = VQC_HardwareEfficientAnsatz(4, ["rx", "RY", "rz"],
                                            entangle_gate="cnot",
                                            entangle_rules="linear",
                                            depth=2)
                self.encode1 = RZ(wires=0)
                self.encode2 = RZ(wires=1)
                self.measure = Probability(wires=[0, 2])
                self.device = TNQMachine(4)
            def forward(self, y, *args, **kwargs):
                self.device.reset_states(self.batch_size)
                self.encode1(params = y[0],q_machine = self.device,)
                self.encode2(params = y[1],q_machine = self.device,)
                self.ansatz(q_machine =self.device)
                return self.measure(q_machine =self.device)

        class QM(Module):
            def __init__(self, batch_size, name=""):
                super().__init__(name)
                self.linearx = Linear(4,2)
                self.qpart = QPart(batch_size)
            def forward(self, x, *args, **kwargs):
                return self.qpart(self.linearx(x))

        bz =3
        inputx = tensor.arange(1.0,bz*4+1).reshape([bz,4])
        inputx.requires_grad= True
        qlayer = QM(bz)
        y = qlayer(inputx)
        y.backward(pyvqnet.tensor.ones_like(y))
        print(y)


VQC_BasicEntanglerTemplate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.VQC_BasicEntanglerTemplate(num_layers=1, num_qubits=1, rotation="RX", initial=None, dtype=None)

    由每个量子位上的单参数单量子位旋转组成的层,后跟一个闭合链或环组合的多个CNOT门。

    CNOT 门环将每个量子位与其邻居连接起来,最后一个量子位被认为是第一个量子位的邻居。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。


    :param num_layers: 量子比特线路层数。
    :param num_qubits: 量子比特数,默认为1。
    :param rotation: 使用单参数单量子比特门,``RX`` 被用作默认值。
    :param initial: 使用initial 初始化所有其中参数逻辑门的参数,默认值: None,此模块将随机初始化参数。
    :param dtype: 参数的数据类型,默认值: None,使用float32。
    :return: 返回一个含可训练参数的VQC_BasicEntanglerTemplate实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import TNQModule, VQC_BasicEntanglerTemplate, Probability, TNQMachine
        from pyvqnet import tensor


        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_BasicEntanglerTemplate(2,
                                                    4,
                                                    "rz",
                                                    initial=tensor.ones([1, 1]))

                self.measure = Probability(wires=[0, 2])
                self.device = TNQMachine(4)

            def forward(self,x, *args, **kwargs):

                self.ansatz(q_machine=self.device)
                return self.measure(q_machine=self.device)

        bz = 1
        inputx = tensor.arange(1.0, bz * 4 + 1).reshape([bz, 4])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward(pyvqnet.tensor.ones_like(y))
        print(y)



VQC_StronglyEntanglingTemplate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.VQC_StronglyEntanglingTemplate(num_layers=1, num_qubits=1, ranges=None, initial=None, dtype=None)

    由单个量子比特旋转和纠缠器组成的层,参考 `circuit-centric classifier design <https://arxiv.org/abs/1804.00633>`__ .

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。


    :param num_layers: 重复层数,默认值: 1。
    :param num_qubits: 量子比特数,默认值: 1。
    :param ranges: 决定每个后续层的范围超参数的序列。默认值: None,对第 :math:`l` 层和 :math:`M` 个量子比特使用 :math:`r=l \mod M`。
    :param initial: 使用initial 初始化所有其中参数逻辑门的参数,默认值: None,此模块将随机初始化参数。
    :param dtype: 参数的数据类型,默认值: None,使用 float32。
    :return: VQC_StronglyEntanglingTemplate 实例


    Example::

        from pyvqnet.nn.torch import TorchModule,Linear,TorchModuleList
        from pyvqnet.qnn.vqc.tn.native.qcircuit import VQC_StronglyEntanglingTemplate
        from pyvqnet.qnn.vqc.tn.native import Probability, TNQMachine, TNQModule
        from pyvqnet import tensor
        import pyvqnet

        pyvqnet.backends.set_backend("pyvqnet")
        pyvqnet.utils.set_random_seed(25)
        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_StronglyEntanglingTemplate(2,
                                                    4,
                                                    None,
                                                    initial=tensor.ones([1, 1]))

                self.measure = Probability(wires=[0, 1])
                self.device = TNQMachine(4)

            def forward(self,x, *args, **kwargs):

                self.ansatz(q_machine=self.device)
                return self.measure(q_machine=self.device)

        bz = 1
        inputx = tensor.arange(1.0, bz * 4 + 1).reshape([bz, 4])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward(pyvqnet.tensor.ones_like(y))
        print(y)

VQC_QuantumEmbedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.native.VQC_QuantumEmbedding(  num_repetitions_input, depth_input, num_unitary_layers, num_repetitions,initial = None,dtype = None,name= "")

    使用 RZ,RY,RZ 创建变分量子电路,将经典数据编码为量子态。
    参考 `Quantum embeddings for machine learning <https://arxiv.org/abs/2001.03622>`_。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。


    :param num_repetitions_input: 子模块中输入编码的重复次数。
    :param depth_input: 输入维数。
    :param num_unitary_layers: 变分量子门的重复次数。
    :param num_repetitions: 子模块的重复次数。
    :param initial: 参数初始化值,默认为None
    :param dtype: 参数的类型,默认 None,使用float32.
    :param name: 类的名字

    Example::

        from pyvqnet.qnn.vqc.tn.native.qcircuit import VQC_QuantumEmbedding
        from pyvqnet.qnn.vqc.tn.native import TNQMachine, MeasureAll, TNQModule
        from pyvqnet import tensor
        import pyvqnet

        pyvqnet.backends.set_backend("pyvqnet")
        pyvqnet.utils.set_random_seed(25)
        depth_input = 2
        num_repetitions = 2
        num_repetitions_input = 2
        num_unitary_layers = 2
        nq = depth_input * num_repetitions_input
        bz = 12

        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_QuantumEmbedding(num_repetitions_input, depth_input,
                                                num_unitary_layers,
                                                num_repetitions, initial=tensor.full([1],12.0),dtype=pyvqnet.kfloat32)

                self.measure = MeasureAll(obs={f"Z{nq-1}":1})
                self.device = TNQMachine(nq)

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(bz)
                self.ansatz(x,q_machine=self.device)
                return self.measure(q_machine=self.device)

        inputx = tensor.arange(1.0, bz * depth_input + 1).reshape([bz, depth_input])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward(pyvqnet.tensor.ones_like(y))
        print(y)


ExpressiveEntanglingAnsatz
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.native.ExpressiveEntanglingAnsatz(type: int, num_wires: int, depth: int, dtype=None, name: str = "")

    论文 `Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms <https://arxiv.org/pdf/1905.10876.pdf>`_ 中的 19 种不同的ansatz。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.native.QModule`` 以及 ``pyvqnet.nn.Module``。
        该类可以作为 ``pyvqnet.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。


    :param type: 电路类型从 1 到 19,共19种线路。
    :param num_wires: 量子比特数。
    :param depth: 电路深度。
    :param dtype: 参数的数据类型, 默认值: None, 使用 float32。
    :param name: 名字,默认"".

    :return:
        一个 ExpressiveEntanglingAnsatz 实例

    Example::

        from pyvqnet.qnn.vqc.tn.native.qcircuit import ExpressiveEntanglingAnsatz
        from pyvqnet.qnn.vqc.tn.native import Probability, TNQMachine, MeasureAll, TNQModule
        from pyvqnet import tensor
        import pyvqnet

        pyvqnet.backends.set_backend("pyvqnet")
        pyvqnet.utils.set_random_seed(25)

        class QModel(TNQModule):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = TNQMachine(num_wires, dtype=dtype)
                self.c1 = ExpressiveEntanglingAnsatz(1,3,2)
                self.measure = MeasureAll(obs={
                    "Z1":1
                })

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(1)
                self.c1(q_machine = self.qm)
                rlt = self.measure(q_machine=self.qm)
                return rlt
            

        input_x = tensor.QTensor([[0.1, 0.2, 0.3]])

        qunatum_model = QModel(num_wires=3, dtype=pyvqnet.kcomplex64)

        batch_y = qunatum_model(input_x)
        batch_y.backward()
        print(batch_y)


vqc_basis_embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.native.vqc_basis_embedding(basis_state,q_machine,wires=None)
   :noindex:

    将n个二进制特征编码到 ``q_machine`` 的n个量子比特的基态。该函数别名 `VQC_BasisEmbedding` 。

    例如, 对于 ``basis_state=([0, 1, 1])``, 在量子系统下其基态为 :math:`|011 \rangle`。

    :param basis_state:  ``(n)`` 大小的二进制输入。
    :param q_machine: 量子虚拟机设备。
    :param wires: 编码作用的量子比特,默认为 None,使用 ``q_machine`` 的全部量子比特。
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import vqc_basis_embedding,TNQMachine
        qm  = TNQMachine(3)
        vqc_basis_embedding(basis_state=[1,1,0],q_machine=qm)
        print(qm.get_states())




vqc_angle_embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.native.vqc_angle_embedding(input_feat, wires, q_machine: pyvqnet.qnn.vqc.tn.native.TNQMachine, rotation: str = "X")
   :noindex:

    将 :math:`N` 特征编码到 :math:`n` 量子比特的旋转角度中, 其中 :math:`N \leq n`。
    该函数别名 `VQC_AngleEmbedding` 。

    旋转可以选择为 : 'X' , 'Y' , 'Z', 如 ``rotation`` 的参数定义为:

    * ``rotation='X'`` 将特征用作RX旋转的角度。

    * ``rotation='Y'`` 将特征用作RY旋转的角度。

    * ``rotation='Z'`` 将特征用作RZ旋转的角度。

     ``wires`` 代表旋转门在量子比特上的idx。

    :param input_feat: 表示参数的数组。
    :param wires: 量子比特idx。
    :param q_machine: 量子虚拟机设备。
    :param rotation: 旋转门,默认为“X”。
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import vqc_angle_embedding, TNQMachine
        from pyvqnet.tensor import QTensor
        qm  = TNQMachine(2)
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='X')
        print(qm.get_states())
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='Y')
        print(qm.get_states())
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='Z')
        print(qm.get_states())


vqc_amplitude_embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.native.vqc_amplitude_embedding(input_feature, q_machine)

    将 :math:`2^n` 特征编码为 :math:`n` 量子比特的振幅向量。该函数别名 `VQC_AmplitudeEmbedding` 。

    :param input_feature: 表示参数的numpy数组。
    :param q_machine: 量子虚拟机设备。
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import vqc_amplitude_embedding, TNQMachine
        from pyvqnet.tensor import QTensor
        qm  = TNQMachine(3)
        vqc_amplitude_embedding(QTensor([3.2,-2,-2,0.3,12,0.1,2,-1]), q_machine=qm)
        print(qm.get_states())



vqc_iqp_embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:function:: pyvqnet.qnn.vqc.tn.native.vqc_iqp_embedding(input_feat, q_machine: pyvqnet.qnn.vqc.tn.native.TNQMachine, rep: int = 1)
   :noindex:

    使用IQP线路的对角门将 :math:`n` 特征编码为 :math:`n` 量子比特。该函数别名:  ``VQC_IQPEmbedding`` 。

    编码是由 `Havlicek et al. (2018) <https://arxiv.org/pdf/1804.11326.pdf>`_ 提出。

    通过指定 ``rep`` ,可以重复基本IQP线路。

    :param input_feat: 表示参数的数组。
    :param q_machine: 量子虚拟机设备。
    :param rep: 重复量子线路块次数,默认次数为1。
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import vqc_iqp_embedding, TNQMachine
        from pyvqnet.tensor import QTensor
        qm  = TNQMachine(3)
        vqc_iqp_embedding(QTensor([3.2,-2,-2]), q_machine=qm)
        print(qm.get_states())        



vqc_rotcircuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.native.vqc_rotcircuit(q_machine, wire, params)
   :noindex:

    任意单量子比特旋转的量子逻辑门组合。该函数别名:  ``VQC_RotCircuit`` 。

    .. math::

        R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
        e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
        e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.


    :param q_machine: 量子虚拟机设备。
    :param wire: 量子比特索引。
    :param params: 表示参数  :math:`[\phi, \theta, \omega]`。
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import vqc_rotcircuit, TNQMachine
        from pyvqnet.tensor import QTensor
        qm  = TNQMachine(3)
        vqc_rotcircuit(q_machine=qm, wire=[1],params=QTensor([2.0,1.5,2.1]))
        print(qm.get_states())


vqc_crot_circuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.native.vqc_crot_circuit(para,control_wire,rot_wire,q_machine)
   :noindex:

	受控Rot单量子比特旋转的量子逻辑门组合。该函数别名:  ``VQC_CRotCircuit`` 。

    .. math:: CR(\phi, \theta, \omega) = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0\\
            0 & 0 & e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2)\\
            0 & 0 & e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.
    
    :param para: 表示参数的数组。
    :param control_wire: 控制量子比特索引。
    :param rot_wire: Rot量子比特索引。
    :param q_machine: 量子虚拟机设备。
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn.native import vqc_crot_circuit,TNQMachine, MeasureAll
        p = QTensor([2, 3, 4.0])
        qm = TNQMachine(2)
        vqc_crot_circuit(p, 0, 1, qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)




vqc_controlled_hadamard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.native.vqc_controlled_hadamard(wires, q_machine)
   :noindex:

    受控Hadamard逻辑门量子线路。该函数别名:  ``VQC_Controlled_Hadamard`` 。

    .. math:: CH = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
            0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
        \end{bmatrix}.

    :param wires: 量子比特索引列表, 第一位是控制比特, 列表长度为2。
    :param q_machine: 量子虚拟机设备。
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn.native import vqc_controlled_hadamard,\
            TNQMachine, MeasureAll

        p = QTensor([0.2, 3, 4.0])
        qm = TNQMachine(3)
        vqc_controlled_hadamard([1, 0], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)



vqc_ccz
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.native.vqc_ccz(wires, q_machine)
   :noindex:

    受控-受控-Z (controlled-controlled-Z) 逻辑门。该函数别名:  ``VQC_CCZ`` 。

    .. math::

        CCZ =
        \begin{pmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & -1
        \end{pmatrix}
    
    :param wires: 量子比特下标列表,第一位是控制比特。列表长度为3。
    :param q_machine: 量子虚拟机设备。
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn.native import vqc_ccz,TNQMachine, MeasureAll
        p = QTensor([0.2, 3, 4.0])

        qm = TNQMachine(3)

        vqc_ccz([1, 0, 2], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)



vqc_fermionic_single_excitation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.native.vqc_fermionic_single_excitation(weight, wires, q_machine)
   :noindex:

    对泡利矩阵的张量积求幂的耦合簇单激励算子。矩阵形式下式给出:

    .. math::

        \hat{U}_{pr}(\theta) = \mathrm{exp} \{ \theta_{pr} (\hat{c}_p^\dagger \hat{c}_r
        -\mathrm{H.c.}) \},

    该函数别名:  ``VQC_FermionicSingleExcitation`` 。

    :param weight:  量子比特p上的参数, 只有一个元素.
    :param wires: 表示区间[r, p]中的量子比特索引子集。最小长度必须为2。第一索引值被解释为r,最后一个索引值被解释为p。
                中间的索引被CNOT门作用,以计算量子位集的奇偶校验。
    :param q_machine: 量子虚拟机设备。

    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn.native import vqc_fermionic_single_excitation,\
            TNQMachine, MeasureAll
        qm = TNQMachine(3)
        p0 = QTensor([0.5])

        vqc_fermionic_single_excitation(p0, [1, 0, 2], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)

 


vqc_fermionic_double_excitation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.native.vqc_fermionic_double_excitation(weight, wires1, wires2, q_machine)
   :noindex:

    对泡利矩阵的张量积求幂的耦合聚类双激励算子,矩阵形式由下式给出:

    .. math::

        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \{ \theta (\hat{c}_p^\dagger \hat{c}_q^\dagger
        \hat{c}_r \hat{c}_s - \mathrm{H.c.}) \},

    其中 :math:`\hat{c}` 和 :math:`\hat{c}^\dagger` 是费米子湮灭和
    创建运算符和索引 :math:`r, s` 和 :math:`p, q` 在占用的和
    分别为空分子轨道。 使用 `Jordan-Wigner 变换
    <https://arxiv.org/abs/1208.5986>`_ 上面定义的费米子算子可以写成
    根据 Pauli 矩阵(有关更多详细信息,请参见
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_)

    .. math::

        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \Big\{
        \frac{i\theta}{8} \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1}
        \hat{Z}_a (\hat{X}_s \hat{X}_r \hat{Y}_q \hat{X}_p +
        \hat{Y}_s \hat{X}_r \hat{Y}_q \hat{Y}_p +\\ \hat{X}_s \hat{Y}_r \hat{Y}_q \hat{Y}_p +
        \hat{X}_s \hat{X}_r \hat{X}_q \hat{Y}_p - \mathrm{H.c.}  ) \Big\}

    该函数别名:  ``VQC_FermionicDoubleExcitation`` 。

    :param weight: 可变参数
    :param wires1: 代表的量子比特的索引列表区间 [s, r] 中占据量子比特的子集。第一个索引被解释为 s,最后一索引被解释为 r。 CNOT 门对中间的索引进行操作,以计算一组量子位的奇偶性。
    :param wires2: 代表的量子比特的索引列表区间 [q, p] 中占据量子比特的子集。第一根索引被解释为 q,最后一索引被解释为 p。 CNOT 门对中间的索引进行操作,以计算一组量子位的奇偶性。
    :param q_machine: 量子虚拟机设备。

    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn.native import vqc_fermionic_double_excitation,\
            TNQMachine, MeasureAll
        qm = TNQMachine(5)
        p0 = QTensor([0.5])

        vqc_fermionic_double_excitation(p0, [0, 1], [2, 3], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)
 

vqc_uccsd
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.native.vqc_uccsd(weights, wires, s_wires, d_wires, init_state, q_machine)
   :noindex:

    实现酉耦合簇单激发和双激发拟设(UCCSD)。UCCSD 是 VQE 拟设,通常用于运行量子化学模拟。

    在一阶 Trotter 近似内,UCCSD 酉函数由下式给出:

    .. math::

        \hat{U}(\vec{\theta}) =
        \prod_{p > r} \mathrm{exp} \Big\{\theta_{pr}
        (\hat{c}_p^\dagger \hat{c}_r-\mathrm{H.c.}) \Big\}
        \prod_{p > q > r > s} \mathrm{exp} \Big\{\theta_{pqrs}
        (\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s-\mathrm{H.c.}) \Big\}

    其中 :math:`\hat{c}` 和 :math:`\hat{c}^\dagger` 是费米子湮灭和
    创建运算符和索引 :math:`r, s` 和 :math:`p, q` 在占用的和
    分别为空分子轨道。(更多细节见
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_):

    该函数别名:  ``VQC_UCCSD`` 。

    :param weights: 包含参数的大小 ``(len(s_wires)+ len(d_wires))`` 张量
        :math:`\theta_{pr}` 和 :math:`\theta_{pqrs}` 输入 Z 旋转
        ``FermionicSingleExcitation`` 和 ``FermionicDoubleExcitation`` 。
    :param wires: 模板作用的量子比特索引
    :param s_wires: 包含量子比特索引的列表序列 ``[r,...,p]``
        由单一激发产生
        :math:`\vert r, p \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF} \rangle`,
        其中 :math:`\vert \mathrm{HF} \rangle` 表示 Hartee-Fock 参考态。
    :param d_wires: 列表序列,每个列表包含两个列表
        指定索引 ``[s, ...,r]`` 和 ``[q,..., p]`` 
        定义双激励 :math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r\hat{c}_s \vert \mathrm{HF} \rangle` 。
    :param init_state: 长度 ``len(wires)`` occupation-number vector 表示
        高频状态。 ``init_state`` 在量子比特初始化状态。
    :param q_machine: 量子虚拟机设备。
    
    
    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import vqc_uccsd, TNQMachine, MeasureAll
        from pyvqnet.tensor import QTensor
        p0 = QTensor([2, 0.5, -0.2, 0.3, -2, 1, 3, 0])
        s_wires = [[0, 1, 2], [0, 1, 2, 3, 4], [1, 2, 3], [1, 2, 3, 4, 5]]
        d_wires = [[[0, 1], [2, 3]], [[0, 1], [2, 3, 4, 5]], [[0, 1], [3, 4]],
                [[0, 1], [4, 5]]]
        qm = TNQMachine(6)

        vqc_uccsd(p0, range(6), s_wires, d_wires, QTensor([1.0, 1, 0, 0, 0, 0]), qm)
        m = MeasureAll(obs={"Z1": 1})
        exp = m(q_machine=qm)
        print(exp)

        # [[0.963802]]


vqc_zfeaturemap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.native.vqc_zfeaturemap(input_feat, q_machine: pyvqnet.qnn.vqc.tn.native.TNQMachine, data_map_func=None, rep: int = 2)
   :noindex:

    一阶泡利 Z 演化电路。

    对于 3 个量子位和 2 次重复,电路表示为:

    .. parsed-literal::

        ┌───┐┌──────────────┐┌───┐┌──────────────┐
        ┤ H ├┤ U1(2.0*x[0]) ├┤ H ├┤ U1(2.0*x[0]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[1]) ├┤ H ├┤ U1(2.0*x[1]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[2]) ├┤ H ├┤ U1(2.0*x[2]) ├
        └───┘└──────────────┘└───┘└──────────────┘
    
    泡利弦固定为 ``Z``。 因此,一阶展开将是一个没有纠缠门的电路。

    :param input_feat: 表示输入参数的数组。
    :param q_machine: 量子虚拟机。
    :param data_map_func: 参数映射矩阵, 为可调用函数, 设计方式为: ``data_map_func = lambda x: x``。
    :param rep: 模块重复次数。
    
    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import vqc_zfeaturemap, TNQMachine, hadamard
        from pyvqnet.tensor import QTensor
        qm = TNQMachine(3)
        for i in range(3):
            hadamard(q_machine=qm, wires=[i])
        vqc_zfeaturemap(input_feat=QTensor([[0.1,0.2,0.3]]),q_machine = qm)
        print(qm.get_states())
 

vqc_zzfeaturemap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.native.vqc_zzfeaturemap(input_feat, q_machine: pyvqnet.qnn.vqc.tn.native.TNQMachine, data_map_func=None, entanglement: Union[str, List[List[int]],Callable[[int], List[int]]] = "full",rep: int = 2)
   :noindex:

    二阶 Pauli-Z 演化电路。

    对于 3 个量子位、1 个重复和线性纠缠,电路表示为:

    .. parsed-literal::

        ┌───┐┌─────────────────┐
        ┤ H ├┤ U1(2.0*φ(x[0])) ├──■────────────────────────────■────────────────────────────────────
        ├───┤├─────────────────┤┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[1])) ├┤ X ├┤ U1(2.0*φ(x[0],x[1])) ├┤ X ├──■────────────────────────────■──
        ├───┤├─────────────────┤└───┘└──────────────────────┘└───┘┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[2])) ├──────────────────────────────────┤ X ├┤ U1(2.0*φ(x[1],x[2])) ├┤ X ├
        └───┘└─────────────────┘                                  └───┘└──────────────────────┘└───┘
    
    其中 ``φ`` 是经典的非线性函数,如果输入两个值则 ``φ(x,y) = (pi - x)(pi - y)``, 输入一个则为 ``φ(x) = x``, 用 ``data_map_func`` 表示如下:
    
    .. code-block::
        
        def data_map_func(x):
            coeff = x if x.shape[-1] == 1 else ft.reduce(lambda x, y: (np.pi - x) * (np.pi - y), x)
            return coeff

    :param input_feat: 表示输入参数的数组。
    :param q_machine: 量子虚拟机。
    :param data_map_func: 参数映射矩阵, 为可调用函数。 
    :param entanglement: 指定的纠缠结构。
    :param rep: 模块重复次数。
    
    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import vqc_zzfeaturemap, TNQMachine
        from pyvqnet.tensor import QTensor

        qm = TNQMachine(3)
        vqc_zzfeaturemap(q_machine=qm, input_feat=QTensor([[0.1,0.2,0.3]]))
        print(qm.get_states())


vqc_allsinglesdoubles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.native.vqc_allsinglesdoubles(weights, q_machine: pyvqnet.qnn.vqc.tn.native.TNQMachine, hf_state, wires, singles=None, doubles=None)
   :noindex:

    在这种情况下,我们有四个单激发和双激发来保留 Hartree-Fock 态的总自旋投影。

    由此产生的酉矩阵保留了粒子数量,并在初始 Hartree-Fock 状态和编码多激发配置的其他状态的叠加中准备了n量子位系统。
      
    :param weights: 大小为 ``(len(singles) + len(doubles),)`` 的QTensor,包含按顺序进入 vqc.qCircuit.single_excitation 和 vqc.qCircuit.double_excitation 操作的角度
    :param q_machine: 量子虚拟机。
    :param hf_state: 代表 Hartree-Fock 状态的长度 ``len(wires)`` 占用数向量, ``hf_state`` 用于初始化线路。
    :param wires: 作用的量子位。
    :param singles: 具有single_exitation操作所作用的两个量子位索引的列表序列。
    :param doubles: 具有double_exitation操作所作用的两个量子位索引的列表序列。

    例如,两个电子和六个量子位情况下的量子电路如下图所示:
    
.. image:: ./images/all_singles_doubles.png
    :width: 600 px
    :align: center

|

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import vqc_allsinglesdoubles, TNQMachine

        from pyvqnet.tensor import QTensor
        qubits = 4
        qm = TNQMachine(qubits)

        vqc_allsinglesdoubles(q_machine=qm, weights=QTensor([0.55, 0.11, 0.53]), 
                              hf_state = QTensor([1,1,0,0]), singles=[[0, 2], [1, 3]], doubles=[[0, 1, 2, 3]], wires=[0,1,2,3])
        print(qm.get_states())

vqc_basisrotation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.native.vqc_basisrotation(q_machine: pyvqnet.qnn.vqc.tn.native.TNQMachine, wires, unitary_matrix: QTensor, check=False)
   :noindex:

    实现一个电路,提供可用于执行精确的单体基础旋转的整体。线路来自于 `arXiv:1711.04789 <https://arxiv.org/abs/1711.04789>`_\ 中给出的单粒子费米子确定的酉变换 :math:`U(u)`
    
    .. math::

        U(u) = \exp{\left( \sum_{pq} \left[\log u \right]_{pq} (a_p^\dagger a_q - a_q^\dagger a_p) \right)}.
    
    :math:`U(u)` 通过使用论文 `Optica, 3, 1460 (2016) <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_\ 中给出的方案。
    

    :param q_machine: 量子虚拟机。
    :param wires: 作用的量子位。
    :param unitary_matrix: 指定基础变换的矩阵。
    :param check: 检测 `unitary_matrix` 是否为酉矩阵。

    Example::

        import pyvqnet

        pyvqnet.backends.set_backend("pyvqnet")
        from pyvqnet.qnn.vqc.tn.native import vqc_basisrotation, TNQMachine
        from pyvqnet.tensor import QTensor
        import numpy as np

        V = np.array([[0.73678 + 0.27511j, -0.5095 + 0.10704j, -0.06847 + 0.32515j],
                    [0.73678 + 0.27511j, -0.5095 + 0.10704j, -0.06847 + 0.32515j],
                    [-0.21271 + 0.34938j, -0.38853 + 0.36497j, 0.61467 - 0.41317j]])

        eigen_vals, eigen_vecs = np.linalg.eigh(V)
        umat = eigen_vecs.T
        wires = range(len(umat))

        qm = TNQMachine(len(umat))

        vqc_basisrotation(q_machine=qm,
                        wires=wires,
                        unitary_matrix=QTensor(umat, dtype=qm.dtype))

        print(qm.get_states())


噪声通道
============================================================

以下接口模拟量子噪声。噪声通道是一条与量子逻辑门调用方式相同的指令(例如 ``depolarizing(q_machine=qm, wire=0, p=0.05)``):它按程序顺序作用在模拟器 **当前** 的态上,不附着于任何一个逻辑门。同一组函数既可以作用在态矢模拟器 ``pyvqnet.qnn.vqc.QMachine`` 上,也可以作用在本页“精确密度矩阵模拟器”一节介绍的精确密度矩阵模拟器上。本节介绍态矢模拟器上的行为。

可用的通道如下。强度参数都必须是 Python 数值,不能作为可训练参数。

.. csv-table::
    :header: "通道", "强度参数", "作用"
    :widths: 22, 32, 46

    "``depolarizing``", "``p``,范围 ``[0, 1]``", "去极化通道 ``E(rho) = (1 - p) rho + p I/2``"
    "``amplitude_damping``", "``gamma``,范围 ``[0, 1]``", "振幅阻尼通道(能量弛豫)"
    "``phase_damping``", "``gamma``,范围 ``[0, 1]``", "相位阻尼通道(相干损耗)"
    "``decoherence``", "``T1``, ``T2``, ``gate_time``", "T1/T2 退相干通道"
    "``bit_flip``", "``p``,范围 ``[0, 1]``", "比特翻转通道 ``E(rho) = (1 - p) rho + p X rho X``"
    "``readout``", "2x2 行随机矩阵 ``matrix``", "读出误差通道"

执行模型
---------------------------------------------------------------

在态矢模拟器上,噪声通道按蒙特卡洛轨迹执行:

* 轨迹: ``reset_states(batchsize)`` 的 batch 维的每一项是一条独立的轨迹,每条轨迹自始至终是一个归一化的纯态。
* 分支: 通道作用时,每条轨迹按各 Kraus 分支的概率抽取一个分支,再把该分支的 Kraus 算子作用到这条轨迹的态矢上;非酉的分支作用之后重新归一化。整个 batch 的分支由一次批量操作同时抽取。
* 分支概率: ``depolarizing`` 与 ``bit_flip`` 的分支概率与态无关,分别是 ``1 - 3p/4, p/4, p/4, p/4`` 与 ``1 - p, p``;其余通道的分支概率只依赖该比特上处于 ``|0>`` 与 ``|1>`` 的概率。
* 随机数: 分支由噪声采样生成器产生的均匀随机数决定,种子由 ``QMachine.set_noise_seed`` 或 ``set_noise_seed`` 设置。
* 结果: 测量接口返回每条轨迹上的结果,例如 ``MeasureAll`` 返回 ``[batchsize, 1]`` (``obs`` 为字典)或 ``[batchsize, len(obs)]`` (``obs`` 为列表)。对 batch 维取平均,得到噪声期望的估计,其统计误差随轨迹数按 ``1/sqrt(batchsize)`` 减小。
* 梯度: 分支的选择不可导,可导的是被选中的 Kraus 算子对态的作用。对轨迹的平均值(对每条轨迹的值是线性的代价)做自动微分,得到的是精确噪声期望梯度的无偏估计,估计的噪声同样随轨迹数减小。代价如果对每条轨迹的期望是非线性的(例如对每条轨迹的期望求平方误差),得到的梯度是有偏的,此时可以使用 ``pyvqnet.qnn.SPSA``。

下面的例子用 1024 条轨迹估计去极化噪声下的期望,并与精确值对比。

Example::

    import numpy as np
    from pyvqnet.qnn.vqc import QMachine, MeasureAll, ry, depolarizing

    # batch 维的每个样本是一条独立的蒙特卡洛轨迹
    qm = QMachine(1)
    qm.set_noise_seed(7)
    qm.reset_states(1024)                 # 1024 条轨迹
    ry(q_machine=qm, wires=0, params=0.4)
    depolarizing(q_machine=qm, wire=0, p=0.05)
    y = MeasureAll(obs={"Z0": 1.0})(q_machine=qm)
    # 每条轨迹一个期望值
    print(y.shape)

    # [1024, 1]

    # 噪声 <Z0>, 标准误差约 1/sqrt(1024)
    print(y.to_numpy().mean())

    # 0.8724892

    # 精确值 (1 - p) * cos(0.4)
    print((1 - 0.05) * np.cos(0.4))

    # 0.8750079443027408

所有通道共用如下约束,违反时抛出 ``ValueError``:

* 强度参数越界,或 ``decoherence`` 的 ``T2`` 大于 ``2 * T1``;
* ``wire`` 超出模拟器的比特范围;
* 模拟器处于伴随梯度(adjoint)模式或定义模式(``set_just_defined(True)``),或是张量网络模拟器、比特重排(分布式)模式的模拟器;在作用过通道的模拟器上再开启 adjoint 模式同样会抛出 ``ValueError``, ``reset_states`` 会清除这一标记。

强度为 0 时(``p == 0``, ``gamma == 0``, ``gate_time == 0``),通道是恒等操作。

depolarizing
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.depolarizing(q_machine, wire, p)

    去极化通道 ``E(rho) = (1 - p) rho + p I/2``,作用在 ``q_machine`` 当前态的 ``wire`` 比特上。Kraus 算子为 ``sqrt(1 - 3p/4) I``、``sqrt(p/4) X``、``sqrt(p/4) Y`` 与 ``sqrt(p/4) Z``; ``p == 1`` 时该比特变为最大混态。

    :param q_machine: 量子虚拟机设备。
    :param wire: 通道作用的量子比特索引。
    :param p: 去极化强度,范围 ``[0, 1]``。
    :return: None。

    Example::

        from pyvqnet.qnn.vqc import QMachine, MeasureAll, ry, depolarizing

        qm = QMachine(1)
        qm.set_noise_seed(7)
        qm.reset_states(1024)                 # 1024 条轨迹
        ry(q_machine=qm, wires=0, params=0.4)
        depolarizing(q_machine=qm, wire=0, p=0.05)
        y = MeasureAll(obs={"Z0": 1.0})(q_machine=qm)
        # 噪声 <Z0>
        print(y.to_numpy().mean())

        # 0.8724892

amplitude_damping
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.amplitude_damping(q_machine, wire, gamma)

    振幅阻尼通道(能量弛豫),作用在 ``q_machine`` 当前态的 ``wire`` 比特上。Kraus 算子为 ``K0 = [[1, 0], [0, sqrt(1 - gamma)]]`` 与 ``K1 = [[0, sqrt(gamma)], [0, 0]]``: ``K1`` 把布居从 ``|1>`` 移到 ``|0>``,通道作用在 ``|0>`` 上是恒等。

    :param q_machine: 量子虚拟机设备。
    :param wire: 通道作用的量子比特索引。
    :param gamma: 阻尼强度,范围 ``[0, 1]``。
    :return: None。

    Example::

        import numpy as np
        from pyvqnet.qnn.vqc import QMachine, MeasureAll, ry, amplitude_damping

        theta, gamma = 0.7, 0.3
        qm = QMachine(1)
        qm.set_noise_seed(3)
        qm.reset_states(20000)                # 20000 条轨迹
        ry(q_machine=qm, wires=0, params=theta)
        amplitude_damping(q_machine=qm, wire=0, gamma=gamma)
        z = MeasureAll(obs={"Z0": 1.0})(q_machine=qm).to_numpy().mean()
        print(z)

        # 0.83527464

        # 精确值
        print(np.cos(theta / 2) ** 2 + (2 * gamma - 1) * np.sin(theta / 2) ** 2)

        # 0.8353895310991419

phase_damping
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.phase_damping(q_machine, wire, gamma)

    相位阻尼通道(相干损耗,不交换布居),作用在 ``q_machine`` 当前态的 ``wire`` 比特上。Kraus 算子为 ``K0 = [[1, 0], [0, sqrt(1 - gamma)]]`` 与 ``K1 = [[0, 0], [0, sqrt(gamma)]]``:布居不变,相干按 ``sqrt(1 - gamma)`` 衰减。

    :param q_machine: 量子虚拟机设备。
    :param wire: 通道作用的量子比特索引。
    :param gamma: 阻尼强度,范围 ``[0, 1]``。
    :return: None。

    Example::

        import numpy as np
        from pyvqnet.qnn.vqc import QMachine, MeasureAll, ry, phase_damping

        theta, gamma = 0.7, 0.3
        qm = QMachine(1)
        qm.set_noise_seed(3)
        qm.reset_states(20000)                # 20000 条轨迹
        ry(q_machine=qm, wires=0, params=theta)
        phase_damping(q_machine=qm, wire=0, gamma=gamma)
        x = MeasureAll(obs={"X0": 1.0})(q_machine=qm).to_numpy().mean()
        z = MeasureAll(obs={"Z0": 1.0})(q_machine=qm).to_numpy().mean()
        # 相干衰减: sqrt(1 - gamma) * sin(theta)
        print(x)

        # 0.5393676
        print(np.sqrt(1 - gamma) * np.sin(theta))

        # 0.5389911872980073

        # 布居不衰减: 仍为 cos(theta)
        print(z)

        # 0.76607466
        print(np.cos(theta))

        # 0.7648421872844885

decoherence
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.decoherence(q_machine, wire, T1, T2, gate_time)

    T1/T2 退相干通道,描述一个持续时间为 ``gate_time`` 的门,作用在 ``q_machine`` 当前态的 ``wire`` 比特上。它实现的映射是 ``rho11 -> exp(-t/T1) rho11`` 与 ``rho01 -> exp(-t/T2) rho01``,其中 ``t = gate_time``。三个 Kraus 算子由振幅阻尼(强度 ``g = 1 - exp(-t/T1)``)与相位阻尼(强度 ``lambda = 1 - exp(-t (2/T2 - 1/T1))``)复合而成,与 pyqpanda3 的 ``decoherence_error`` 在相同的 ``T1``, ``T2`` 和门时长下是同一个通道。

    :param q_machine: 量子虚拟机设备。
    :param wire: 通道作用的量子比特索引。
    :param T1: 能量弛豫时间,必须为正数。
    :param T2: 去相干时间,必须为正数且不大于 ``2 * T1``。
    :param gate_time: 门的持续时间,必须是非负数。
    :return: None。

    Example::

        import numpy as np
        from pyvqnet.qnn.vqc import QMachine, MeasureAll, ry, x, decoherence

        T1, T2, gate_time = 20e-6, 10e-6, 5e-6
        # |+> 态上相干按 exp(-t/T2) 衰减
        qm = QMachine(1)
        qm.set_noise_seed(12)
        qm.reset_states(20000)                # 20000 条轨迹
        ry(q_machine=qm, wires=0, params=np.pi / 2)
        decoherence(q_machine=qm, wire=0, T1=T1, T2=T2, gate_time=gate_time)
        print(MeasureAll(obs={"X0": 1.0})(q_machine=qm).to_numpy().mean())

        # 0.6066284
        print(np.exp(-gate_time / T2))

        # 0.6065306597126334

        # |1> 态上布居按 exp(-t/T1) 衰减
        qm = QMachine(1)
        qm.set_noise_seed(11)
        qm.reset_states(20000)                # 20000 条轨迹
        x(q_machine=qm, wires=0)
        decoherence(q_machine=qm, wire=0, T1=T1, T2=T2, gate_time=gate_time)
        print(MeasureAll(obs={"Z0": 1.0})(q_machine=qm).to_numpy().mean())

        # -0.5621
        print(1 - 2 * np.exp(-gate_time / T1))

        # -0.5576015661428098

bit_flip
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.bit_flip(q_machine, wire, p)

    比特翻转通道 ``E(rho) = (1 - p) rho + p X rho X``,作用在 ``q_machine`` 当前态的 ``wire`` 比特上。它给出对称读出矩阵 ``[[1 - p, p], [p, 1 - p]]`` 在计算基下的测量统计; ``p == 0`` 是恒等通道。应当在计算基测量之前紧邻作用。

    :param q_machine: 量子虚拟机设备。
    :param wire: 通道作用的量子比特索引。
    :param p: 翻转概率,范围 ``[0, 1]``。
    :return: None。

    Example::

        import numpy as np
        from pyvqnet.qnn.vqc import QMachine, MeasureAll, ry, bit_flip

        theta, p = 0.7, 0.2
        qm = QMachine(1)
        qm.set_noise_seed(21)
        qm.reset_states(20000)                # 20000 条轨迹
        ry(q_machine=qm, wires=0, params=theta)
        bit_flip(q_machine=qm, wire=0, p=p)
        print(MeasureAll(obs={"Z0": 1.0})(q_machine=qm).to_numpy().mean())

        # 0.45936415

        # 精确值
        print((1 - 2 * p) * np.cos(theta))

        # 0.4589053123706931

readout
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.readout(q_machine, wire, matrix, at_measurement=False)

    单个被测比特上的读出误差通道。``matrix[b][o] = P(读出 o | 真实 b)`` 是 2x2 的行随机矩阵:元素在 ``[0, 1]`` 内且每行之和为 1。它实现为 measure-and-prepare 通道,Kraus 算子为 ``K_{b->o} = sqrt(matrix[b][o]) |o><b|``,因此测量统计与对真实的计算基结果施加经典读出误差一致。

    ``at_measurement=False`` (默认)时通道立即作用在当前态上。``at_measurement=True`` 时不改变当前态,而是把读出延迟到下一次测量:

    * ``Probability`` 与 ``Samples`` 把矩阵折算进它们使用的概率,态保持不变;
    * 其他测量(例如 ``MeasureAll``)在测量边界把通道作用到态上。

    延迟的读出在下一次测量调用时被消费:未被该次测量覆盖的比特上的读出被丢弃,之后的测量看到的是未受影响的分布。``reset_states`` 会清空尚未消费的延迟读出。延迟读出假定这次调用与测量之间没有其他逻辑门。

    .. warning::

        单位矩阵在这里是完全退相干通道,不是恒等映射;对称矩阵 ``[[1 - p, p], [p, 1 - p]]`` 请使用 ``bit_flip``。

    :param q_machine: 量子虚拟机设备。
    :param wire: 被测的量子比特索引。
    :param matrix: 2x2 的行随机矩阵, ``matrix[b][o] = P(读出 o | 真实 b)``。
    :param at_measurement: 是否把读出延迟到下一次测量,默认值 False。
    :return: None。

    Example::

        import numpy as np
        from pyvqnet.qnn.vqc import QMachine, MeasureAll, Probability, Samples, readout, x
        from pyvqnet.utils import set_random_seed

        M = [[0.9, 0.1], [0.3, 0.7]]          # M[真实][读出]

        # 立即作用: |1> 以 0.3 的概率被读成 0
        qm = QMachine(1)
        qm.set_noise_seed(23)
        qm.reset_states(20000)                # 20000 条轨迹
        x(q_machine=qm, wires=0)
        readout(q_machine=qm, wire=0, matrix=M)
        print(MeasureAll(obs={"Z0": 1.0})(q_machine=qm).to_numpy().mean())

        # -0.3982

        # 精确值
        print(M[1][0] - M[1][1])

        # -0.39999999999999997

        # 延迟到测量: 态不变, 矩阵折算进概率; 延迟的读出被这一次测量消费
        qm = QMachine(1)
        qm.reset_states(1)
        x(q_machine=qm, wires=0)
        readout(q_machine=qm, wire=0, matrix=M, at_measurement=True)
        print(Probability(wires=[0])(q_machine=qm).to_numpy())

        # [[0.3 0.7]]
        readout(q_machine=qm, wire=0, matrix=M, at_measurement=True)
        set_random_seed(7)
        samples = Samples(wires=[0], shots=4000)(q_machine=qm).to_numpy()
        # 被读成 0 的比例, 约 0.3
        print(np.mean(samples == 0))

        # 0.3035

set_noise_seed
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.common.noise.set_noise_seed(seed)

    设置进程级的噪声分支采样器种子,作用于没有调用过 ``QMachine.set_noise_seed`` 的模拟器。采样器按后端与设备各维护一条随机流;每次调用都重新播种,因此以同一个种子重新调用可以复现,否则两次运行之间的随机流会继续前进。

    :param seed: ``pyvqnet.random.Generator`` 的种子; ``None`` 表示使用新的随机熵。
    :return: None。

apply_channel
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.common.noise.apply_channel(qm, name, wire, value)

    按通道名作用一个通道,是各个通道函数的通用入口。``value`` 是该通道自己的强度参数: ``depolarizing`` 与 ``bit_flip`` 为 ``p``, ``amplitude_damping`` 与 ``phase_damping`` 为 ``gamma``, ``decoherence`` 为 ``(T1, T2, gate_time)``, ``readout`` 为读出矩阵。

    :param qm: 量子虚拟机设备。
    :param name: 通道名,取本节通道表中的名字。
    :param wire: 通道作用的量子比特索引。
    :param value: 通道的强度参数。
    :return: None。未知的通道名抛出 ``ValueError``。

    Example::

        import numpy as np
        from pyvqnet.qnn.vqc import QMachine, MeasureAll, ry
        from pyvqnet.qnn.vqc.common.noise import apply_channel

        qm = QMachine(1)
        qm.set_noise_seed(3)
        qm.reset_states(20000)                # 20000 条轨迹
        ry(q_machine=qm, wires=0, params=0.7)
        apply_channel(qm, "amplitude_damping", 0, 0.3)
        print(MeasureAll(obs={"Z0": 1.0})(q_machine=qm).to_numpy().mean())

        # 0.83527464

        # 精确值
        print(np.cos(0.35) ** 2 + (2 * 0.3 - 1) * np.sin(0.35) ** 2)

        # 0.8353895310991419

kraus_builder
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.common.noise.kraus_builder(name)

    返回名为 ``name`` 的通道的 Kraus 算子构造器。构造器的调用形式是 ``constructor(value, dtype)``,返回该通道在强度 ``value`` 与数据类型 ``dtype`` 下的 Kraus 算子(``QTensor``)列表;精确密度矩阵模拟器用这些 Kraus 算子构造通道的超算符。

    :param name: 通道名,取本节通道表中的名字。
    :return: 通道的 Kraus 算子构造器。

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.qnn.vqc.common.noise import kraus_builder

        # kraus_builder(name) 返回该通道 Kraus 算子的构造器: constructor(强度参数, dtype) -> Kraus 算子列表
        build = kraus_builder("depolarizing")
        for kraus in build(0.2, pyvqnet.kcomplex64):
            print(np.round(kraus.to_numpy(), 4))

        # [[0.922+0.j 0.   +0.j]
        #  [0.   +0.j 0.922+0.j]]
        # [[0.    +0.j 0.2236+0.j]
        #  [0.2236+0.j 0.    +0.j]]
        # [[ 0.+0.j     -0.-0.2236j]
        #  [ 0.+0.2236j  0.+0.j    ]]
        # [[ 0.2236+0.j  0.    +0.j]
        #  [ 0.    +0.j -0.2236+0.j]]

含噪声的变分线路训练
---------------------------------------------------------------

噪声通道对态的作用在自动微分图中,所以含噪声的变分线路可以像无噪声时一样训练: ``forward`` 开头用 ``reset_states`` 指定轨迹数,线路中插入通道,对返回的期望取平均作为损失。下面的例子在去极化噪声下训练 ``RY`` 的参数,把两个比特制备到 ``|11>``,损失是 ``<Z0> + <Z1>``。噪声下的精确最优损失是 ``((1 - p) + (1 - p)^2) * cos(theta)`` (第二个比特上的噪声作用在已经被第一个比特的噪声削弱过的态上),无噪声时是 ``2 * cos(theta)``。

.. code-block::

    import numpy as np
    from pyvqnet.nn import Module
    from pyvqnet.optim.sgd import SGD
    from pyvqnet.qnn.vqc import QMachine, RY, CNOT, MeasureAll, depolarizing
    from pyvqnet.tensor import QTensor

    P = 0.1


    class NoisyModel(Module):
        # 去极化噪声下把两个比特制备到 |11>: 损失为 <Z0> + <Z1>

        def __init__(self):
            super().__init__()
            self.qm = QMachine(2)
            self.qm.set_noise_seed(42)
            self.theta = RY(trainable=True, wires=0, init_params=QTensor([[0.2]]))
            self.cnot = CNOT(wires=[0, 1])
            self.measure = MeasureAll(obs={"Z0": 1.0, "Z1": 1.0})

        def forward(self, x, *args, **kwargs):
            self.qm.reset_states(x.shape[0])            # x.shape[0] 是轨迹数
            self.theta(q_machine=self.qm)
            depolarizing(q_machine=self.qm, wire=0, p=P)
            self.cnot(q_machine=self.qm)
            depolarizing(q_machine=self.qm, wire=1, p=P)
            return self.measure(q_machine=self.qm)


    model = NoisyModel()
    optimizer = SGD(model.parameters(), lr=0.5)
    trajectories = QTensor(np.zeros((4096, 1), np.float32))   # 4096 条轨迹

    for step in range(21):
        if step % 5 == 0:
            print(step, model(trajectories).to_numpy().mean())
        optimizer.zero_grad()
        model(trajectories).mean().backward()
        optimizer.step()

    # 0 1.6423771
    # 5 -1.6083918
    # 10 -1.7026367
    # 15 -1.7158203
    # 20 -1.7021484

    theta = float(model.theta.params.to_numpy().reshape(-1)[0])
    print(theta)

    # 3.1415927410125732

    # 精确的噪声损失
    print(((1 - P) + (1 - P) ** 2) * np.cos(theta))

    # -1.7099999999999935

    # 无噪声时的损失
    print(2 * np.cos(theta))

    # -1.9999999999999925

测量接口在噪声下的返回
---------------------------------------------------------------

* ``MeasureAll(obs, name="")`` 没有采样次数参数,返回每条轨迹上的解析期望: ``obs`` 为字典时形状是 ``[batchsize, 1]``,为列表时是 ``[batchsize, len(obs)]``。对 batch 维取平均得到噪声期望的估计。
* ``Probability(wires, name="")`` 返回每条轨迹在 ``wires`` 上的边缘概率,不做采样;在 ``readout(..., at_measurement=True)`` 之后调用时,把读出矩阵折算进概率。
* ``Samples(wires=None, obs=None, shots=1, name="")`` 对每个样本,从它当前的概率分布中有放回地抽取 ``shots`` 个样本,返回 int64 的 ``[batchsize, shots, len(wires)]``。随机数来自后端的默认随机数生成器: ``pyvqnet.utils.set_random_seed(seed)`` 重置该随机流,之后的第一次采样结果相同,连续调用则依次前进。在 ``readout(..., at_measurement=True)`` 之后调用时,先把读出矩阵折算进概率再采样。


噪声策略
============================================================

噪声策略把一组通道规则安装到模拟器上,由引擎在匹配的逻辑门之后(``after_gate``)以及每次测量之前(``before_measurement``)自动作用通道,与在相同位置手工调用通道函数逐位等价。一台模拟器最多安装一个策略。本节的类与函数从 ``pyvqnet.qnn.vqc.common.noise_policy`` 导入。

策略中可用的通道是 ``depolarizing``, ``decoherence``, ``bit_flip`` 与 ``readout``。

.. warning::

    策略依赖逐门注入通道的执行路径。张量网络模拟器,以及 C++ 录制器已打开的模拟器不能使用启用的策略: ``install`` 抛出 ``RuntimeError``,或在此后的 ``reset_states`` 处抛出 ``RuntimeError``。态在 CUDA 设备上的态矢模拟器在 ``reset_states`` 时会打开该录制,因此噪声策略只在 CPU 的态矢模拟器上可用。``fused_*`` 系列融合门不经过逐门注入的入口,在安装了启用策略的模拟器上调用它们会抛出 ``RuntimeError``。

NoiseRule
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.common.noise_policy.NoiseRule(channel, params, gate_names=("*",), qubits=("*",), position="after_gate")

    一条通道规则: 把一个通道作用到与它匹配的逻辑门(``position="after_gate"``)或测量(``position="before_measurement"``)上。不可变的 dataclass。

    :param channel: 通道名, ``"depolarizing"``, ``"decoherence"``, ``"bit_flip"`` 或 ``"readout"``。
    :param params: 通道强度参数的字典,键必须与通道完全一致: ``depolarizing`` 与 ``bit_flip`` 是 ``{"p"}``, ``decoherence`` 是 ``{"T1", "T2", "gate_time"}``, ``readout`` 是 ``{"matrix"}``。
    :param gate_names: 匹配的逻辑门名,字符串或字符串的序列; ``"*"`` (默认值)匹配所有门,比较时不区分大小写。
    :param qubits: 匹配的量子比特,整数、``"*"`` 或由整数与 ``"*"`` 组成的序列; ``"*"`` (默认值)匹配所有比特。
    :param position: 规则的位置, ``"after_gate"`` (默认值)或 ``"before_measurement"``。

    构造时即校验: 不支持的通道或位置, ``params`` 中缺失或未知的键,越界的强度参数,类型不符的 ``gate_names`` 或 ``qubits`` 都抛出 ``ValueError``。构造之后 ``gate_names`` 被存为小写的元组, ``qubits`` 被存为元组, ``params`` 中的数值被存为浮点数。

    ``after_gate`` 规则在每个逻辑门执行之后应用: 规则的 ``gate_names`` 匹配该门时,对门涉及的每一个比特,若 ``qubits`` 也匹配,就在该比特上作用通道。定义模式下的门,以及测量 X/Y 分量时插入的对角化旋转门不触发规则。

    ``before_measurement`` 规则在每次测量之前、对测量的比特应用一次: ``Samples`` 与 ``Probability`` 是它们的 ``wires``; ``MeasureAll`` 是观测量中带 ``Z`` 的比特,且要求观测量中非零的项都只由 ``Z`` 与 ``I`` 组成,否则不注入通道;其他测量不注入通道。

    Example::

        from pyvqnet.qnn.vqc.common.noise_policy import NoiseRule

        rule = NoiseRule(channel="depolarizing", params={"p": 0.02}, gate_names="RY", qubits=[0, 2])
        # 门名转为小写, qubits 规范化为元组
        print(rule)

        # NoiseRule(channel='depolarizing', params={'p': 0.02}, gate_names=('ry',), qubits=(0, 2), position='after_gate')
        try:
            NoiseRule(channel="depolarizing", params={"p": 0.02}, position="before_gate")
        except ValueError as error:
            print(error)

        # unsupported noise policy position 'before_gate'; expected one of ['after_gate', 'before_measurement']

NoisePolicy
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.common.noise_policy.NoisePolicy(rules, enabled=True)

    一个有序的规则集合(不可变的 dataclass)。门与测量的入口按规则顺序依次应用其中启用的规则。

    :param rules: ``NoiseRule`` 的序列,按顺序应用;可以为空。
    :param enabled: 为 ``False`` 时策略被安装但不注入任何通道,默认值 ``True``。

    .. py:method:: to_json()

        把策略序列化为 JSON 字符串。

        :return: JSON 字符串。

    .. py:method:: from_dict(data)

        类方法。从字典构造策略,是严格的解析器: 未知的键、缺失的 ``"rules"``、不是 ``bool`` 的 ``"enabled"`` 都抛出 ``ValueError``。``"rules"`` 是规则的列表,每条规则是只含 ``"channel"``, ``"params"``, ``"gate_names"``, ``"qubits"``, ``"position"`` 的映射,其中 ``"channel"`` 与 ``"params"`` 必填。

        :param data: 策略的字典。
        :return: NoisePolicy。

    .. py:method:: from_json(text)

        类方法。从 JSON 字符串构造策略,格式要求与 ``from_dict`` 相同。

        :param text: JSON 字符串。
        :return: NoisePolicy。

    Example::

        import numpy as np
        from pyvqnet.qnn.vqc import QMachine, Probability, depolarizing, readout, ry, cnot
        from pyvqnet.qnn.vqc.common.noise_policy import NoisePolicy, NoiseRule, install, uninstall, active_config

        policy = NoisePolicy(rules=(
            NoiseRule(channel="depolarizing", params={"p": 0.02}, gate_names="ry"),
            NoiseRule(channel="readout", params={"matrix": ((0.85, 0.15), (0.25, 0.75))},
                      position="before_measurement"),
        ))

        # 策略自动注入: 每个 ry 之后、每次测量之前
        qm = QMachine(2)
        qm.set_noise_seed(13)
        install(qm, policy)
        qm.reset_states(512)                   # 512 条轨迹
        ry(q_machine=qm, wires=0, params=0.3)
        cnot(q_machine=qm, wires=[0, 1])
        ry(q_machine=qm, wires=1, params=0.5)
        auto = Probability(wires=[0, 1])(q_machine=qm).to_numpy()
        print(active_config(qm) is policy)

        # True
        uninstall(qm)

        # 等价的手工写法: 在相同位置显式调用通道
        qm = QMachine(2)
        qm.set_noise_seed(13)
        qm.reset_states(512)                   # 512 条轨迹
        ry(q_machine=qm, wires=0, params=0.3)
        depolarizing(q_machine=qm, wire=0, p=0.02)
        cnot(q_machine=qm, wires=[0, 1])
        ry(q_machine=qm, wires=1, params=0.5)
        depolarizing(q_machine=qm, wire=1, p=0.02)
        readout(q_machine=qm, wire=0, matrix=((0.85, 0.15), (0.25, 0.75)))
        readout(q_machine=qm, wire=1, matrix=((0.85, 0.15), (0.25, 0.75)))
        manual = Probability(wires=[0, 1])(q_machine=qm).to_numpy()
        print(np.array_equal(auto, manual))

        # True
        print(auto.mean(axis=0))

        # [0.64453125 0.15625    0.14648438 0.05273438]
        print(policy.to_json())

        # {"enabled": true, "rules": [{"channel": "depolarizing", "gate_names": ["ry"], "params": {"p": 0.02}, "position": "after_gate", "qubits": ["*"]}, {"channel": "readout", "gate_names": ["*"], "params": {"matrix": [[0.85, 0.15], [0.25, 0.75]]}, "position": "before_measurement", "qubits": ["*"]}]}
        print(NoisePolicy.from_json(policy.to_json()) == policy)

        # True

install
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.common.noise_policy.install(q_machine, config)

    把策略 ``config`` 安装到 ``q_machine`` 上,替换此前安装的策略。此后的逻辑门与测量按 ``config`` 注入通道。

    :param q_machine: 量子虚拟机设备。
    :param config: 要安装的 ``NoisePolicy``。
    :return: None。 ``config`` 处于启用状态而该模拟器的执行路径会绕过策略时抛出 ``RuntimeError``。

uninstall
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.common.noise_policy.uninstall(q_machine)

    卸载 ``q_machine`` 上的策略,此后的逻辑门与测量不再注入通道。

    :param q_machine: 量子虚拟机设备。
    :return: None。

active_config
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.common.noise_policy.active_config(q_machine)

    返回 ``q_machine`` 上安装的 ``NoisePolicy``,没有安装时返回 ``None``。

    :param q_machine: 量子虚拟机设备。
    :return: 已安装的 ``NoisePolicy`` 或 ``None``。

validate_execution_config
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.common.noise_policy.validate_execution_config(q_machine)

    检查 ``q_machine`` 上安装的策略能否被它的执行路径执行: 没有安装策略或策略未启用时什么都不做;策略已启用而执行路径会绕过它时抛出 ``RuntimeError``。逻辑门与测量的入口在执行前都会调用它。

    :param q_machine: 量子虚拟机设备。
    :return: None。

    Example::

        from pyvqnet.qnn.vqc import QMachine
        from pyvqnet.qnn.vqc.common.noise_policy import (
            NoisePolicy, NoiseRule, install, uninstall, active_config, validate_execution_config,
        )

        qm = QMachine(2)
        # 未安装策略
        print(active_config(qm))

        # None

        policy = NoisePolicy(rules=(NoiseRule(channel="bit_flip", params={"p": 0.01}),))
        install(qm, policy)                   # 替换此前安装的策略
        print(active_config(qm) is policy)

        # True
        validate_execution_config(qm)         # 这台模拟器能执行该策略: 不抛出

        install(qm, NoisePolicy(rules=policy.rules, enabled=False))
        print(active_config(qm).enabled)

        # False
        uninstall(qm)
        print(active_config(qm))

        # None


精确密度矩阵模拟器
============================================================

``pyvqnet.qnn.vqc.dm`` 提供精确的密度矩阵模拟器: 它直接演化密度矩阵,噪声通道作为超算符整体作用,没有采样误差。该包需要显式导入。

* ``pyvqnet.qnn.vqc.dm.native``: ``pyvqnet`` 后端的 ``QMachine`` (别名 ``DensityMatrixQMachine``),并重新导出量子逻辑门的函数与模块(``ry``, ``CNOT`` 等)以及 ``MeasureAll``, ``Probability``, ``Samples``, ``QInfoMeasurement``。
* ``pyvqnet.qnn.vqc.dm.torch``: ``torch`` 后端的同名导出,使用前需要执行 ``pyvqnet.backends.set_backend("torch")``。
* 噪声通道函数(``depolarizing`` 等)两个后端共用,从 ``pyvqnet.qnn.vqc`` 导入。

模拟器的 ``states`` 是形状为 ``[batch, 2**n, 2**n]`` 的密度矩阵 ``rho``: 行下标是 ket 一侧,列下标是 bra 一侧。``reset_states(batchsize)`` 把它重置为 ``|0...0><0...0|`` 并复制 ``batchsize`` 份。与态矢模拟器一样,batch 维用于批量数据(每个样本可以有自己的角度);噪声不再需要轨迹, ``batch`` 也就不再表示轨迹数。

模拟器有两条执行路径:

* 逐门执行: 逻辑门按 ``rho -> U rho U^\dagger`` 逐个作用,用普通的自动微分(``backward()``)求梯度。这是直接运行模型时的路径。
* 分块(tile)路径: 把 ``QMachine`` 所在的模型包进 ``QuantumLayerAdjoint`` 之后,引擎先录制线路,再把 ``rho`` 展开成 ``2n`` 个比特上的态矢,交给分块执行器完成前向与反向;噪声通道作为通道记录,与门交错编入同一个执行计划。是否走这条路径由环境变量 ``VQNET_DM_MODE`` 控制。

态包含 ``4**n`` 个复数元素。比特数没有固定的上限,只受内存限制: ``complex64`` 下每个样本 ``n = 8`` 约 0.5 MB, ``n = 12`` 约 134 MB, ``n = 14`` 约 2 GB。

QMachine(pyvqnet 后端)
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.dm.native.QMachine(num_wires, dtype=pyvqnet.kcomplex64, save_ir=False, state_storage=None)

    ``pyvqnet`` 后端的精确密度矩阵模拟器。 ``pyvqnet.qnn.vqc.dm.native.DensityMatrixQMachine`` 是它的别名。 ``states`` 属性是形状为 ``[batch, 2**num_wires, 2**num_wires]`` 的密度矩阵。

    .. note::

        与态矢模拟器一样,每次运行完整的量子线路之前必须调用 ``reset_states(batchsize)``。密度矩阵不参与 ``state_dict`` 的保存与载入。

    :param num_wires: 量子比特数。
    :param dtype: 计算数据类型, ``pyvqnet.kcomplex64`` (默认值)或 ``pyvqnet.kcomplex128``,对应的参数精度分别是 ``pyvqnet.kfloat32`` 与 ``pyvqnet.kfloat64``。
    :param save_ir: 是否保存 OriginIR,默认值 False。逐门执行的逻辑门不写入 OriginIR 与 ``op_history``,因此该开关只生成 ``QINIT``/``CREG`` 头。
    :param state_storage: 态的存储数据类型,默认值 None(与 ``dtype`` 相同)。传入 ``pyvqnet.kcomplex32`` 时态以 complex32 存储,计算仍在 complex64 下进行。
    :return: 精确密度矩阵模拟器。

    Example::

        import numpy as np
        from pyvqnet.qnn.vqc import MeasureAll, Probability, Samples, QInfoMeasurement
        from pyvqnet.qnn.vqc.dm.native import QMachine, DensityMatrixQMachine, ry, cnot
        from pyvqnet.utils import set_random_seed

        print(DensityMatrixQMachine is QMachine)

        # True

        qm = QMachine(2)                         # 精确密度矩阵模拟器
        qm.reset_states(3)
        # [batch, 2**n, 2**n]
        print(qm.states.shape, qm.states.dtype)

        # [3, 4, 4] complex64

        ry(q_machine=qm, wires=0, params=np.pi / 2)
        cnot(q_machine=qm, wires=[0, 1])         # 贝尔态
        print(np.round(qm.states.to_numpy()[0], 4))

        # [[0.5+0.j 0. +0.j 0. +0.j 0.5+0.j]
        #  [0. +0.j 0. +0.j 0. +0.j 0. +0.j]
        #  [0. +0.j 0. +0.j 0. +0.j 0. +0.j]
        #  [0.5+0.j 0. +0.j 0. +0.j 0.5+0.j]]

        print(MeasureAll(obs={"Z0": 1.0, "Z1": 1.0})(q_machine=qm).to_numpy())

        # [[0.]
        #  [0.]
        #  [0.]]

        print(Probability(wires=[0, 1])(q_machine=qm).to_numpy()[0])

        # [0.49999997 0.         0.         0.49999997]

        set_random_seed(11)
        samples = Samples(wires=[0, 1], shots=4)(q_machine=qm)
        # [batch, shots, len(wires)]
        print(samples.shape)

        # [3, 4, 2]

        print(samples.to_numpy()[0])

        # [[0 0]
        #  [1 1]
        #  [0 0]
        #  [1 1]]

        # 比特 0 的约化态是最大混态, 纯度为 0.5
        print(QInfoMeasurement("purity", wires=[0])(q_machine=qm).to_numpy())

        # [0.49999994 0.49999994 0.49999994]

QMachine(torch 后端)
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.dm.torch.QMachine(num_wires, dtype=pyvqnet.kcomplex64, save_ir=False, state_storage=None)

    ``torch`` 后端的精确密度矩阵模拟器,继承自 ``pyvqnet.nn.torch.TorchModule``。 ``pyvqnet.qnn.vqc.dm.torch.DensityMatrixQMachine`` 是它的别名。构造参数与 ``pyvqnet.qnn.vqc.dm.native.QMachine`` 相同。 ``states`` 与测量结果是 ``torch.Tensor``,梯度由 ``torch`` 的自动微分计算。

    Example::

        import numpy as np
        import torch
        import pyvqnet
        from pyvqnet.qnn.vqc import depolarizing          # 噪声通道由两个后端共用

        saved = pyvqnet.backends.get_backend_name()
        pyvqnet.backends.set_backend("torch")
        import pyvqnet.qnn.vqc.dm.torch as dm_torch

        theta = torch.tensor([[0.7]], dtype=torch.float64, requires_grad=True)
        qm = dm_torch.QMachine(1, dtype=pyvqnet.kcomplex128)
        qm.reset_states(1)
        dm_torch.ry(q_machine=qm, wires=0, params=theta)
        depolarizing(q_machine=qm, wire=0, p=0.2)
        y = dm_torch.MeasureAll(obs={"Z0": 1.0})(q_machine=qm)
        print(type(y).__name__, y.dtype, y.detach().numpy())

        # Tensor torch.float64 [[0.61187375]]
        y.sum().backward()
        print(theta.grad.item())

        # -0.5153741497901528
        print(-(1 - 0.2) * np.sin(0.7))

        # -0.5153741497901528
        pyvqnet.backends.set_backend(saved)

密度矩阵模拟器上的量子逻辑门与编码
---------------------------------------------------------------

``pyvqnet.qnn.vqc`` 中的量子逻辑门接口(函数形式与模块形式)可以直接作用在精确密度矩阵模拟器上,按 ``rho -> U rho U^\dagger`` 演化; ``use_dagger`` 与逐样本的角度同样有效。振幅编码(例如 ``VQC_AmplitudeEmbedding``)把纯态的振幅写成 ``|psi><psi|``。

密度矩阵模拟器上的测量
---------------------------------------------------------------

* ``MeasureAll(obs, name="")`` 返回 Pauli 和的精确期望,形状是 ``[batch, 1]`` (``obs`` 为字典)或 ``[batch, len(obs)]`` (``obs`` 为列表)。 ``Z`` 项直接由 ``rho`` 的对角元计算, ``X``/``Y`` 项通过对角化旋转之后的边缘概率计算;测量之后 ``rho`` 保持不变。
* ``Probability(wires, name="")`` 返回 ``rho`` 对角元的联合边缘概率,形状是 ``[batch, 2**len(wires)]``。
* ``Samples(wires=None, obs=None, shots=1, name="")`` 从对角元的分布中采样,形状是 ``[batch, shots, len(wires)]``,随机数同样由 ``pyvqnet.utils.set_random_seed`` 控制。
* ``QInfoMeasurement(quantity, wires=None, obs=None, name="")`` 的 ``"purity"``, ``"entropy"``, ``"mutual_info"``, ``"variance"``, ``"meyer_wallach"`` 与 ``"expectation"`` 都可以直接读取密度矩阵,其中 ``"expectation"`` 计算的是 ``Tr(M rho)``。
* ``ShadowExpval`` 与 ``DerandomizedShadowExpval`` 建立在纯态采样之上,在密度矩阵模拟器上抛出 ``ValueError``。

下面的例子在贝尔态的一个比特上作用去极化通道得到混合态,再读取几个量子信息量。

.. code-block::

    import numpy as np
    import pyvqnet
    from pyvqnet.qnn.vqc import QInfoMeasurement, depolarizing
    from pyvqnet.qnn.vqc.dm.native import QMachine, ry, cnot, hadamard
    from pyvqnet.tensor import QTensor

    qm = QMachine(2, dtype=pyvqnet.kcomplex128)
    qm.reset_states(1)
    ry(q_machine=qm, wires=0, params=np.pi / 2)
    cnot(q_machine=qm, wires=[0, 1])               # 贝尔态
    depolarizing(q_machine=qm, wire=1, p=0.3)      # 混合态: 0.7 |Bell><Bell| + 0.3 I/4
    print(QInfoMeasurement("purity", wires=[0, 1])(q_machine=qm).to_numpy())

    # [[0.6175]]
    print(0.7 ** 2 + 2 * 0.7 * 0.3 / 4 + 0.3 ** 2 / 4)

    # 0.6174999999999999

    # 自然对数
    print(QInfoMeasurement("entropy", wires=[0])(q_machine=qm).to_numpy())

    # [0.69314718]
    print(QInfoMeasurement("mutual_info", wires=([0], [1]))(q_machine=qm).to_numpy())

    # [0.60594276]

    qm = QMachine(2, dtype=pyvqnet.kcomplex128)
    qm.reset_states(1)
    hadamard(q_machine=qm, wires=0)
    Z = QTensor(np.diag([1.0, -1.0]), dtype=pyvqnet.kcomplex128)
    print(QInfoMeasurement("expectation", obs={"wires": [0], "observables": Z})(q_machine=qm).to_numpy())

    # [[0.]]
    print(QInfoMeasurement("variance", wires=[0], obs=Z)(q_machine=qm).to_numpy())

    # [[1.]]

密度矩阵模拟器上的噪声通道
---------------------------------------------------------------

噪声通道函数与态矢模拟器上相同(``depolarizing``, ``amplitude_damping``, ``phase_damping``, ``decoherence``, ``bit_flip``, ``readout``),但在密度矩阵上作用的是通道的精确映射: 先由 Kraus 算子构造超算符 ``S = sum_k K_k ⊗ conj(K_k)`` (按后端、通道名、强度、数据类型与设备缓存),再把 ``S`` 作为 ``{wire, wire + num_wires}`` 这两个比特上的 4x4 矩阵作用于 ``vec(rho)``。因此 ``batch`` 维不需要轨迹,测量结果就是精确的噪声期望,对 ``rho`` 的梯度也是精确的。

精确值与态矢模拟器的采样结果对比如下。

.. code-block::

    import numpy as np
    from pyvqnet.qnn.vqc import QMachine as StateVectorQMachine, MeasureAll, depolarizing, ry
    from pyvqnet.qnn.vqc.dm.native import QMachine

    theta, p = 0.7, 0.2

    # 精确密度矩阵: 通道作为超算符整体作用, 没有采样误差
    qm = QMachine(1)
    qm.reset_states(1)
    ry(q_machine=qm, wires=0, params=theta)
    depolarizing(q_machine=qm, wire=0, p=p)
    print(MeasureAll(obs={"Z0": 1.0})(q_machine=qm).to_numpy())

    # [[0.6118738]]

    # 态矢模拟器: 每个样本是一条轨迹, 对轨迹取平均
    for trajectories in (200, 2000, 20000):
        sv = StateVectorQMachine(1)
        sv.set_noise_seed(11)
        sv.reset_states(trajectories)
        ry(q_machine=sv, wires=0, params=theta)
        depolarizing(q_machine=sv, wire=0, p=p)
        print(MeasureAll(obs={"Z0": 1.0})(q_machine=sv).to_numpy().mean())

    # 0.6042252
    # 0.61569786
    # 0.6151625

    # 精确值
    print((1 - p) * np.cos(theta))

    # 0.6118737498275908

``readout(..., at_measurement=True)`` 在密度矩阵模拟器上不做折算: 任何测量都在测量边界把它作为精确通道作用到态上。``NoisePolicy`` 在逐门执行时同样可用。

.. code-block::

    import numpy as np
    from pyvqnet.qnn.vqc import (MeasureAll, Probability, depolarizing, amplitude_damping,
                                  bit_flip, readout, ry, x)
    from pyvqnet.qnn.vqc.dm.native import QMachine

    theta, p, gamma = 0.7, 0.2, 0.3
    M = ((0.9, 0.1), (0.3, 0.7))


    def z_after(channel):
        qm = QMachine(1)
        qm.reset_states(1)
        ry(q_machine=qm, wires=0, params=theta)
        channel(qm)
        return MeasureAll(obs={"Z0": 1.0})(q_machine=qm).to_numpy()[0, 0]


    print(z_after(lambda qm: depolarizing(q_machine=qm, wire=0, p=p)), (1 - p) * np.cos(theta))

    # 0.6118738 0.6118737498275908
    print(z_after(lambda qm: amplitude_damping(q_machine=qm, wire=0, gamma=gamma)),
          np.cos(theta / 2) ** 2 + (2 * gamma - 1) * np.sin(theta / 2) ** 2)

    # 0.8353895 0.8353895310991419
    print(z_after(lambda qm: bit_flip(q_machine=qm, wire=0, p=p)), (1 - 2 * p) * np.cos(theta))

    # 0.45890525 0.4589053123706931
    print(z_after(lambda qm: readout(q_machine=qm, wire=0, matrix=M)),
          np.cos(theta / 2) ** 2 * (M[0][0] - M[0][1]) + np.sin(theta / 2) ** 2 * (2 * M[1][0] - 1))

    # 0.6589054 0.6589053123706932

    # 延迟读出在密度矩阵上作为精确通道作用在测量边界
    qm = QMachine(1)
    qm.reset_states(1)
    x(q_machine=qm, wires=0)
    readout(q_machine=qm, wire=0, matrix=M, at_measurement=True)
    print(Probability(wires=[0])(q_machine=qm).to_numpy())

    # [[0.3 0.7]]

    # 态被改变: 0.3|0><0| + 0.7|1><1|
    print(np.round(qm.states.to_numpy()[0].real, 4))

    # [[0.3 0. ]
    #  [0.  0.7]]

密度矩阵模拟器上的梯度
---------------------------------------------------------------

逐门执行时,门与通道都在自动微分图中,直接对输出调用 ``backward()`` 即可得到精确的梯度。

.. code-block::

    import numpy as np
    import pyvqnet
    from pyvqnet.qnn.vqc import MeasureAll, depolarizing
    from pyvqnet.qnn.vqc.dm.native import QMachine, ry
    from pyvqnet.tensor import QTensor, tensor


    def noisy_z(theta):
        qm = QMachine(1, dtype=pyvqnet.kcomplex128)
        qm.reset_states(1)
        ry(q_machine=qm, wires=0, params=theta)
        depolarizing(q_machine=qm, wire=0, p=0.2)
        return MeasureAll(obs={"Z0": 1.0})(q_machine=qm)


    theta = QTensor([[0.7]], dtype=pyvqnet.kfloat64, requires_grad=True)
    y = noisy_z(theta)
    tensor.sums(y).backward()
    print(theta.grad.to_numpy())

    # [[-0.51537415]]

    step = 1e-5                            # 中心差分对照
    up = noisy_z(QTensor([[0.7 + step]], dtype=pyvqnet.kfloat64)).to_numpy()
    down = noisy_z(QTensor([[0.7 - step]], dtype=pyvqnet.kfloat64)).to_numpy()
    print((up - down) / (2 * step))

    # [[-0.51537415]]
    print(-(1 - 0.2) * np.sin(0.7))

    # -0.5153741497901528

把模型包进 ``QuantumLayerAdjoint`` 之后,密度矩阵线路可以通过分块路径求梯度。录制时每个门写成 ket 与 bra 两条记录: ket 一侧作用在比特 ``w`` 上,bra 一侧作用在镜像比特 ``w + n`` 上、矩阵是 ``conj(U)``,最终得到的 ``2n`` 比特态矢就是 ``vec(rho)``;测量成为它的伴随行;噪声通道作为通道记录与门交错编入同一个执行计划,反向回卷时 ket 一侧乘通道超算符的逆,bra 一侧乘它的伴随。

.. warning::

    不能直接把密度矩阵模拟器置于 adjoint 模式( ``reset_states`` 会抛出 ``ValueError``),密度矩阵的伴随梯度只能通过 ``QuantumLayerAdjoint`` 使用。

``VQNET_DM_MODE`` 控制密度矩阵模拟器在 ``QuantumLayerAdjoint`` 中的执行方式,每次前向时读取,需要在前向之前设置。

.. csv-table::
    :header: "取值", "含义"
    :widths: 14, 86

    "``0``", "只逐门执行,分块路径不运行。"
    "``1``", "默认值(未设置时)。优先走分块路径;某一步被分块路径拒绝时,整层改为逐门执行,数值与梯度不变。"
    "``2``", "只允许分块路径。某一步被拒绝时抛出 ``RuntimeError``,错误信息给出被拒绝的原因。"

取 ``0``, ``1``, ``2`` 之外的值时抛出 ``ValueError``。 ``VQNET_ADJOINT_MODE`` 只控制态矢模拟器,不影响密度矩阵模拟器。在 ``pyvqnet.no_grad()`` 下的前向不需要梯度,即使在取值 ``2`` 下也不算被拒绝。

分块路径拒绝某一步的常见原因:

* 门不在分块执行器的字节码表内。2.18.3 的函数形式中, ``iswap``, ``isingxy``, ``cswap``, ``controlledphaseshift``, ``multicontrolledx``, ``multi_control_rz``, ``qft`` 与 ``unitary`` 属于这种情况。
* 测量的输出不可导(例如 ``Samples``),没有伴随行。
* 通道的超算符奇异或病态,无法求逆回卷,例如 ``readout`` 以及 ``gamma == 1`` 的 ``amplitude_damping``。
* 模拟器上安装了启用的 ``NoisePolicy``。分块路径只录制门与通道,不录制策略规则,此时策略在逐门执行的回退中生效。

带有噪声通道的一步默认在每次前向时重新录制,并提示 ``the tile plan cache cannot replay this layer``。通道强度如果是常量,可以设置 ``VQNET_ADJOINT_ASSUME_LITERAL_ANGLES=1``,此后这一步可以回放缓存的执行计划。

.. code-block::

    import os
    import numpy as np
    import pyvqnet
    from pyvqnet.qnn.vqc import MeasureAll, QuantumLayerAdjoint
    from pyvqnet.qnn.vqc.dm.native import QMachine, ry, cnot, rz, iswap
    from pyvqnet.tensor import QTensor

    THETA, PHI = 0.7, -0.4


    class Model(pyvqnet.nn.Module):
        # RY(0) -> CNOT(0,1) -> RZ(1)

        def __init__(self):
            super().__init__()
            self.qm = QMachine(2, dtype=pyvqnet.kcomplex128)
            self.theta = QTensor([[THETA]], dtype=pyvqnet.kfloat64, requires_grad=True)
            self.phi = QTensor([[PHI]], dtype=pyvqnet.kfloat64, requires_grad=True)
            self.measure = MeasureAll(obs={"Z0": 1.0, "X0 X1": 0.5})

        def forward(self, x):
            self.qm.reset_states(x.shape[0])
            ry(q_machine=self.qm, wires=0, params=self.theta)
            cnot(q_machine=self.qm, wires=[0, 1])
            rz(q_machine=self.qm, wires=1, params=self.phi)
            return self.measure(q_machine=self.qm)


    x = QTensor(np.zeros((1, 1), np.float64))
    # VQNET_DM_MODE 依次取 0, 1, 2: 数值与梯度相同, 执行方式不同
    for mode in ("0", "1", "2"):
        os.environ["VQNET_DM_MODE"] = mode
        model = Model()
        y = QuantumLayerAdjoint(model)(x)
        y.sum().backward()
        print(mode, y.to_numpy(), model.qm.last_execution_mode,
              model.theta.grad.to_numpy(), model.phi.grad.to_numpy())

    # 0 [[1.06152408]] eager [[-0.29198453]] [[0.12543509]]
    # 1 [[1.06152408]] tile [[-0.29198453]] [[0.12543509]]
    # 2 [[1.06152408]] tile [[-0.29198453]] [[0.12543509]]

    # 精确的期望
    print(np.cos(THETA) + 0.5 * np.sin(THETA) * np.cos(PHI))

    # 1.0615240789651823

    # 精确的梯度
    print(-np.sin(THETA) + 0.5 * np.cos(THETA) * np.cos(PHI),
          -0.5 * np.sin(THETA) * np.sin(PHI))

    # -0.29198453459989515 0.12543509192500715


    class IswapModel(Model):
        # iSWAP 不在分块字节码表内: 这一步会被分块路径拒绝

        def forward(self, x):
            self.qm.reset_states(x.shape[0])
            ry(q_machine=self.qm, wires=0, params=self.theta)
            iswap(q_machine=self.qm, wires=[0, 1])
            return self.measure(q_machine=self.qm)


    # 被拒绝的步: 取值 1 时整层回退到逐门执行, 取值 2 时抛出 RuntimeError
    for mode in ("1", "2"):
        os.environ["VQNET_DM_MODE"] = mode
        model = IswapModel()
        try:
            y = QuantumLayerAdjoint(model)(x)
            result = f"{y.to_numpy()} {model.qm.last_execution_mode}"
        except RuntimeError as error:
            result = str(error)
        print(mode, result)

    # 1 [[1.]] eager
    # 2 VQNET_DM_MODE=2 but the tile forward declined: the gate plan cannot be compiled for the tile bytecode (gate 2 'iswap' is outside the table)
    os.environ.pop("VQNET_DM_MODE")

密度矩阵模拟器不支持的操作
---------------------------------------------------------------

* ``fused_*`` 系列融合门: 它们按态矢的布局编写,调用时抛出 ``FusedKernelUnavailable``,模拟器的状态保持不变。请使用标准的逻辑门接口。
* ``ShadowExpval`` 与 ``DerandomizedShadowExpval``: 抛出 ``ValueError``。
* 直接开启 adjoint 模式( ``reset_states`` 抛出 ``ValueError``),以及定义模式( ``set_just_defined(True)``)下的逻辑门与噪声通道(调用时抛出 ``ValueError``)。分块路径所需的录制只在 ``QuantumLayerAdjoint`` 内部进行。
* 张量网络模式、比特重排(分布式)模式与分布式通信: ``reset_states`` 抛出 ``ValueError``。

GPU 支持
---------------------------------------------------------------

把模拟器(或整个模型)用 ``toGPU`` 放到 CUDA 设备之后,态矢模拟器上的蒙特卡洛噪声通道、有限采样工具,以及精确密度矩阵模拟器的逐门执行与分块路径都可以在 GPU 上运行; ``torch`` 后端的密度矩阵模拟器同样支持 GPU。噪声策略在 GPU 上的密度矩阵模拟器中可用,并使 ``QuantumLayerAdjoint`` 回退到逐门执行。

噪声采样器的随机流按后端与设备区分,同一种子在 CPU 与 GPU 上不保证得到同一组轨迹。

.. code-block::

    import numpy as np
    import pyvqnet
    from pyvqnet.qnn.vqc import MeasureAll, QMachine as StateVectorQMachine, depolarizing, ry
    from pyvqnet.qnn.vqc.dm.native import QMachine
    from pyvqnet.tensor import QTensor, tensor

    # 态矢模拟器: 蒙特卡洛轨迹
    sv = StateVectorQMachine(1)
    sv.toGPU(pyvqnet.DEV_GPU_0)
    sv.set_noise_seed(11)
    sv.reset_states(20000)                # 20000 条轨迹
    ry(q_machine=sv, wires=0, params=0.7)
    depolarizing(q_machine=sv, wire=0, p=0.2)
    print(sv.states.is_cuda, MeasureAll(obs={"Z0": 1.0})(q_machine=sv).to_numpy().mean())

    # True 0.612868

    # 精确密度矩阵模拟器: 前向与自动微分梯度
    theta = QTensor([[0.7]], dtype=pyvqnet.kfloat32, requires_grad=True)
    qm = QMachine(1)
    qm.toGPU(pyvqnet.DEV_GPU_0)
    qm.reset_states(1)
    ry(q_machine=qm, wires=0, params=theta)
    depolarizing(q_machine=qm, wire=0, p=0.2)
    y = MeasureAll(obs={"Z0": 1.0})(q_machine=qm)
    tensor.sums(y).backward()
    print(qm.states.is_cuda, y.to_numpy(), theta.grad.to_numpy())

    # True [[0.6118738]] [[-0.5153742]]
    print((1 - 0.2) * np.cos(0.7), -(1 - 0.2) * np.sin(0.7))

    # 0.6118737498275908 -0.5153741497901528
