实用函数
######################



随机种子生成
*******************************************

set_random_seed
==============================

.. py:function:: pyvqnet.utils.set_random_seed(seed)

    设定全局随机种子。该种子同时作用于 VQNet 后端的随机数引擎和 numpy 的全局随机数发生器。

    :param seed: 随机数种子。

    .. note::
            当指定固定随机数种子时,随机分布将依据随机种子产生固定的伪随机分布。
            影响包括: `tensor.randu` , `tensor.randn` ,含参经典神经网络以及量子计算层的参数初始化。
            设定种子之后, ``numpy.random`` 的取值(例如 ``numpy.random.rand`` )也由同一个种子决定(传给 numpy 时对 :math:`2^{32}` 取模)。
            种子只设定随机序列的起点:连续两次调用 ``tensor.randn`` 得到的是序列中前后相邻的两段,而不是相同的值。

    Example::

        import pyvqnet.tensor as tensor
        from pyvqnet.nn import Parameter
        from pyvqnet.utils import get_random_seed, set_random_seed
        from pyvqnet.utils.initializer import (he_normal, he_uniform, xavier_normal, xavier_uniform,
                                               uniform, quantum_uniform, normal)
        import numpy as np

        set_random_seed(256)

        rn = tensor.randn([2, 3])
        print(rn)
        rn = tensor.randn([2, 3])
        print(rn)
        rn = tensor.randu([2, 3])
        print(rn)

        print("########################################################")
        print(Parameter(shape=[2, 3], initializer=he_normal))
        print(Parameter(shape=[2, 3], initializer=he_uniform))
        print(Parameter(shape=[2, 3], initializer=xavier_normal))
        print(Parameter(shape=[2, 3], initializer=xavier_uniform))
        print(Parameter(shape=[2, 3], initializer=uniform))
        print(Parameter(shape=[2, 3], initializer=quantum_uniform))
        print(Parameter(shape=[2, 3], initializer=normal))

        print(np.random.rand(3))
        set_random_seed(256)
        print(np.random.rand(3))

        # [[-0.9345335, 0.3424167,-1.8297402],
        #  [-1.4597528, 0.7203974,-1.6204817]]
        # <QTensor [2, 3] DEV_CPU float32>

        # [[-0.5145091,-0.2607963, 0.94709  ],
        #  [ 0.6033011, 0.8305053, 0.0332184]]
        # <QTensor [2, 3] DEV_CPU float32>

        # [[0.9713831,0.276447 ,0.7735571],
        #  [0.650921 ,0.6008579,0.377464 ]]
        # <QTensor [2, 3] DEV_CPU float32>
        # ########################################################

        # [[ 0.0785991,-0.1033378, 0.142875 ],
        #  [ 0.3002529,-0.3641986,-1.187332 ]]
        # <Parameter [2, 3] DEV_CPU float32>

        # [[ 0.3981348, 0.2644825, 0.14002  ],
        #  [-0.0720839,-0.1814943, 0.273384 ]]
        # <Parameter [2, 3] DEV_CPU float32>

        # [[-0.7293476,-1.0812973, 0.6525353],
        #  [-0.0754549, 0.8831711, 0.7223919]]
        # <Parameter [2, 3] DEV_CPU float32>

        # [[-0.4537744,-0.2803605, 0.6620895],
        #  [-1.0098293,-0.3035473,-0.9165617]]
        # <Parameter [2, 3] DEV_CPU float32>

        # [[0.7496392,0.8396614,0.4048045],
        #  [0.3368871,0.7443931,0.612397 ]]
        # <Parameter [2, 3] DEV_CPU float32>

        # [[2.7304947,5.205479 ,5.3462524],
        #  [2.5277855,2.6376216,5.137311 ]]
        # <Parameter [2, 3] DEV_CPU float32>

        # [[-0.1683057,-0.6449004, 0.6106159],
        #  [ 1.4165658,-0.7055426,-0.8774387]]
        # <Parameter [2, 3] DEV_CPU float32>
        # [0.0457838 0.5861207 0.2032399]
        # [0.0457838 0.5861207 0.2032399]

get_random_seed
==============================

.. py:function:: pyvqnet.utils.get_random_seed()

    获取当前随机数种子,即最近一次传给 ``set_random_seed`` 的值。

    Example::

        import pyvqnet.tensor as tensor
        from pyvqnet.utils import get_random_seed, set_random_seed

        set_random_seed(256)
        print(get_random_seed())
        rn = tensor.randn([2, 3])
        print(rn)
        set_random_seed(256)
        print(tensor.randn([2, 3]))

        # 256

        # [[-0.9345335, 0.3424167,-1.8297402],
        #  [-1.4597528, 0.7203974,-1.6204817]]
        # <QTensor [2, 3] DEV_CPU float32>

        # [[-0.9345335, 0.3424167,-1.8297402],
        #  [-1.4597528, 0.7203974,-1.6204817]]
        # <QTensor [2, 3] DEV_CPU float32>
