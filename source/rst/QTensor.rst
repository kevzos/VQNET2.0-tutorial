.. _qtensor_api:

QTensor 模块
###########################

VQNet量子机器学习所使用的数据结构QTensor的python接口介绍。QTensor支持常用的多维张量的操作,例如创建函数,数学函数,逻辑函数,矩阵变换等。



QTensor 的函数与属性
******************************************

.. py:class:: pyvqnet.tensor.tensor.QTensor(data, requires_grad=False, device=pyvqnet.DEV_CPU, dtype=None)

    具有动态计算图构造和自动微分的张量。

    :param data: 输入数据,可以是 _core.Tensor 或numpy 数组。
    :param requires_grad: 是否应该跟踪张量的梯度,默认为 False。
    :param device: 储存在哪个设备上,默认: pyvqnet.DEV_CPU,在CPU上。
    :param dtype: 参数的数据类型,默认: None,使用输入数据的类型(由 Python 列表创建的浮点张量使用默认数据类型 kfloat32);
        可取的值见 ``dtype`` 属性,半精度类型 ``kfloat16`` 与 ``kcomplex32`` 见「半精度数据类型」一节。
    :return: 输出 QTensor。

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet.dtype import *
        import numpy as np

        t1 = QTensor(np.ones([2, 3], dtype=np.float32))
        t2 = QTensor([2, 3, 4j, 5])
        t3 = QTensor([[[2, 3, 4, 5], [2, 3, 4, 5]]], dtype=kbool)
        t4 = QTensor([1.5, 2.5, -0.125], dtype=kfloat16)
        t5 = QTensor([1 + 2j, 3 - 1j], dtype=kcomplex32)
        print(t1.dtype, t2.dtype, t3.dtype)
        print(t1)
        print(t2)
        print(t3)
        print(t4.dtype, t4.element_size())
        print(t5.dtype, t5.element_size())

        # float32 complex64 bool

        # [[1.,1.,1.],
        #  [1.,1.,1.]]
        # <QTensor [2, 3] DEV_CPU float32>

        # [2.+0.j,3.+0.j,0.+4.j,5.+0.j]
        # <QTensor [4] DEV_CPU complex64>

        # [[[ True, True, True, True],
        #   [ True, True, True, True]]]
        # <QTensor [1, 2, 4] DEV_CPU bool>
        # float16 2
        # complex32 4

    .. py:attribute:: ndim

        返回张量的维度的个数。
            
        :return: 张量的维度的个数。

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.ndim)

            # 1
    
 

    .. py:attribute:: shape

        返回张量的维度
        
        :return: 一个列表存有张量的维度

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.shape)

            # [4]

 

    .. py:attribute:: size

        返回张量的元素个数。
        
        :return: 张量的元素个数。

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.size)

            # 4


    .. py:method:: numel

        返回张量的元素个数。
        
        :return: 张量的元素个数。

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.numel())

            # 4



    .. py:attribute:: dtype

        返回张量的数据类型。

        支持数据类型如下：

            =========================================  ===============================
            dtype                                      description
            =========================================  ===============================
            ``pyvqnet.kbool``                          布尔变量
            ``pyvqnet.kuint8``                         8比特整数 (无符号)
            ``pyvqnet.kint8``                          8比特整数 (有符号)
            ``pyvqnet.kint16``                         16比特整数 (有符号)
            ``pyvqnet.kint32``                         32比特整数 (有符号)
            ``pyvqnet.kint64``                         64比特整数 (有符号)
            ``pyvqnet.kfloat16``                       16比特半精度浮点数,每个元素占 2 字节
            ``pyvqnet.kfloat32``                       32比特浮点数, 见 https://en.wikipedia.org/wiki/IEEE_754
            ``pyvqnet.kfloat64``                       64比特浮点数, 见 https://en.wikipedia.org/wiki/IEEE_754
            ``pyvqnet.kcomplex32``                     32比特复数，由两个 `float16` 构成,每个元素占 4 字节
            ``pyvqnet.kcomplex64``                     64比特复数，由两个 `float32` 构成
            ``pyvqnet.kcomplex128``                    128比特复数，由两个 `float64` 构成
            ``pyvqnet.kbfloat16``                      16比特浮点数, 有时被称为脑浮点格式（Brain floating point），其位分配为 1 位符号位、8 位指数位、7 位尾数位
            =========================================  ===============================

        ``str(dtype)`` 给出类型的名字,例如 ``float16`` 和 ``complex32``。``element_size()`` 返回每个元素占用的字节数。

        :return: 张量的数据类型。

        Example::

            from pyvqnet.tensor import QTensor
            from pyvqnet.dtype import *
            import numpy as np

            a = QTensor([2.0, 3.0, 4.0, 5.0])
            print(a.dtype)

            for d in (kbool, kint32, kfloat32, kfloat64, kbfloat16, kfloat16, kcomplex32, kcomplex64, kcomplex128):
                t = QTensor([1], dtype=d)
                print(str(d), t.element_size())

            # float32
            # bool 1
            # int32 4
            # float32 4
            # float64 8
            # bfloat16 2
            # float16 2
            # complex32 4
            # complex64 8
            # complex128 16

    .. py:attribute:: is_contiguous

        是否是contiguous的多维数组。

        :return: 如果是contiguous, 返回True, 否则返回False。

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([[2, 3, 4, 5],[2, 3, 4, 5]])
            b = a.is_contiguous
            print(b)
            #True
            c= a.permute((1,0))
            print(c.is_contiguous)
            #False

 

    .. py:method:: zero_grad()

        将张量的梯度设置为零。将在优化过程中被优化器使用。

        :return: 无。

        Example::

            from pyvqnet.tensor import QTensor
            t3 = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            t3.zero_grad()
            print(t3.grad)
            # [0., 0., 0., 0.]
        


    .. py:method:: backward(grad=None)

        利用反向传播算法, 计算当前张量所在的计算图中的所有需计算梯度的张量的梯度。

        :return: 无

        Example::

            from pyvqnet.tensor import QTensor

            target = QTensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0.2]], requires_grad=True)
            y = 2*target + 3
            y.backward()
            print(target.grad)
            #[[2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]]

 
    .. py:method:: to_numpy()

        将张量的数据拷贝到一个新的 numpy.ndarray 里面。

        ``kfloat16`` 张量返回 ``numpy.float16`` 数组; ``kcomplex32`` 张量返回 ``numpy.complex64`` 数组(numpy 没有 complex32 类型)。

        :return: 一个新的 numpy.ndarray 包含 QTensor 数据

        .. note::

            numpy不支持bfloat16类型，需要先转成其他numpy支持的数据类型例如float32，再调用该接口。
            零拷贝的 ``numpy()`` 方法不支持 ``kcomplex32`` 张量,会抛出 ``RuntimeError``,请改用 ``to_numpy()``。

        Example::

            from pyvqnet import kcomplex32
            from pyvqnet.tensor import QTensor
            t3 = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            t4 = t3.to_numpy()
            print(t4)
            print(type(t4).__name__, t4.dtype)

            a = QTensor([1 + 2j, 3 - 1j], dtype=kcomplex32)
            print(a.to_numpy().dtype)
            try:
                a.numpy()      # zero-copy view of the storage: not possible for complex32
            except RuntimeError as e:
                print("RuntimeError:", e)

            # [2. 3. 4. 5.]
            # ndarray float32
            # complex64
            # RuntimeError: complex32 has no numpy zero-copy format; use to_numpy() (upcasts to complex64).

    .. py:method:: item()

        从只包含单个元素的 QTensor 返回唯一的元素。

        :return: 元素值

        Example::

            from pyvqnet.tensor import tensor

            t = tensor.ones([1])
            print(t.item())

            # 1.0


 
    .. py:method:: contiguous()

        返回当前QTensor的contiguous形式 ,如果已经是contiguous, 则返回自身。

        :return: 返回当前QTensor的contiguous形式 ,如果已经是contiguous, 则返回自身。

        Example::

            from pyvqnet.tensor import tensor

            t = tensor.ones([1])
            print(t.contiguous())




    .. py:method:: argmax(*kargs)

        返回输入 QTensor 中所有元素的最大值的索引, 或返回 QTensor 按某一维度的最大值的索引。

        :param dim: 计算argmax的轴, 只接受单个维度。 如果 dim == None, 则返回输入张量中所有元素的最大值的索引。有效的 dim 范围是 [-R, R), 其中 R 是输入的 ndim。 当 dim < 0 时, 它的工作方式与 dim + R 相同。
        :param keepdim: 输出 QTensor 是否保留了最大值索引操作的轴, 默认是False。

        :return: 输入 QTensor 中最大值的索引。

        当 ``dim`` 为 ``None`` 且张量不含元素时,抛出 ``ValueError``。

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            a = QTensor([[1.3398, 0.2663, -0.2686, 0.2450],
                         [-0.7401, -0.8805, -0.3402, -1.1936],
                         [0.4907, -1.3948, -1.0691, -0.3132],
                         [-1.6092, 0.5419, -0.2993, 0.3195]])
            flag = a.argmax()
            print(flag)

            flag_0 = a.argmax([0], True)
            print(flag_0)

            flag_1 = a.argmax([1], True)
            print(flag_1)

            try:
                tensor.argmax(QTensor([]))
            except ValueError as e:
                print("ValueError:", e)

            # [0]
            # <QTensor [1] DEV_CPU int64>

            # [[0,3,0,3]]
            # <QTensor [1, 4] DEV_CPU int32>

            # [[0],
            #  [2],
            #  [0],
            #  [1]]
            # <QTensor [4, 1] DEV_CPU int32>
            # ValueError: argmax(): cannot reduce a tensor with zero elements

    .. py:method:: argmin(*kargs)

        返回输入 QTensor 中所有元素的最小值的索引, 或返回 QTensor 按某一维度的最小值的索引。

        :param dim: 计算argmin的轴, 只接受单个维度。 如果 dim == None, 则返回输入张量中所有元素的最小值的索引。有效的 dim 范围是 [-R, R), 其中 R 是输入的 ndim。 当 dim < 0 时, 它的工作方式与 dim + R 相同。
        :param keepdim: 输出 QTensor 是否保留了最小值索引操作的轴, 默认是False。

        :return: 输入 QTensor 中最小值的索引。

        当 ``dim`` 为 ``None`` 且张量不含元素时,抛出 ``ValueError``。

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            a = QTensor([[1.3398, 0.2663, -0.2686, 0.2450],
                         [-0.7401, -0.8805, -0.3402, -1.1936],
                         [0.4907, -1.3948, -1.0691, -0.3132],
                         [-1.6092, 0.5419, -0.2993, 0.3195]])
            flag = a.argmin()
            print(flag)

            flag_0 = a.argmin([0], True)
            print(flag_0)

            flag_1 = a.argmin([1], False)
            print(flag_1)

            try:
                tensor.argmin(QTensor([]))
            except ValueError as e:
                print("ValueError:", e)

            # [12]
            # <QTensor [1] DEV_CPU int64>

            # [[3,2,2,1]]
            # <QTensor [1, 4] DEV_CPU int32>

            # [2,3,1,0]
            # <QTensor [4] DEV_CPU int32>
            # ValueError: argmin(): cannot reduce a tensor with zero elements

    .. py:method:: fill_(v)

        为当前张量填充特定值, 该函数改变原张量的内部数据。

        :param v: 填充值。

        :return: 无。

        .. note::

            该操作为原地操作。如果张量的 ``requires_grad`` 为 ``True``,自动微分不会记录这次修改,并发出 ``UserWarning``,已有的计算图可能失效;此时请改用非原地的操作。

        Example::

            import warnings
            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            shape = [2, 3]
            value = 42
            t = tensor.zeros(shape)
            t.fill_(value)
            print(t)

            x = tensor.ones([2, 3])
            x.requires_grad = True
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                x.fill_(2.0)
            print(caught[0].category.__name__)
            print(caught[0].message)
            print(x)

            # [[42.,42.,42.],
            #  [42.,42.,42.]]
            # <QTensor [2, 3] DEV_CPU float32>
            # UserWarning
            # fill_ is an in-place operation on a tensor that requires grad; autograd does not record this mutation. Use the out-of-place operation instead.

            # [[2.,2.,2.],
            #  [2.,2.,2.]]
            # <QTensor [2, 3] DEV_CPU float32>

    .. py:method:: all()

        判断张量内数据是否全为非零。

        :return: 返回True, 如果全为非0;否则返回False。

        Example::

            from pyvqnet.tensor import tensor

            shape = [2, 3]
            t = tensor.zeros(shape)
            t.fill_(1.0)
            flag = t.all()
            print(flag)

            # True

    .. py:method:: any()

        判断张量内数据是否有任意元素不为0。

        :return: 返回True, 如果有任意元素不为0;否则返回False。

        Example::

            from pyvqnet.tensor import tensor

            shape = [2, 3]
            t = tensor.ones(shape)
            t.fill_(1.0)
            flag = t.any()
            print(flag)

            # True


    .. py:method:: fill_rand_binary_(v=0.5)

        用从二项分布中随机采样的值填充 QTensor 。

        如果二项分布后随机生成的数据大于二值化阈值 v , 则设置 QTensor 对应位置的元素值为1, 否则为0。

        :param v: 二值化阈值, 默认0.5。

        :return: 无。

        .. note::

            该操作为原地操作。如果张量的 ``requires_grad`` 为 ``True``,自动微分不会记录这次修改,并发出 ``UserWarning``;此时请改用非原地的操作。

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            from pyvqnet.utils import set_random_seed
            import numpy as np
            set_random_seed(42)
            a = np.arange(6).reshape(2, 3).astype(np.float32)
            t = QTensor(a)
            t.fill_rand_binary_(0.5)
            print(t)

            # [[1.,1.,1.],
            #  [1.,0.,0.]]
            # <QTensor [2, 3] DEV_CPU float32>

    .. py:method:: fill_rand_signed_uniform_(v=1)

        用从有符号均匀分布中随机采样的值填充 QTensor 。用缩放因子 v 对生成的随机采样的值进行缩放。

        :param v: 缩放因子, 默认1。

        :return: 无。

        .. note::

            该操作为原地操作。如果张量的 ``requires_grad`` 为 ``True``,自动微分不会记录这次修改,并发出 ``UserWarning``;此时请改用非原地的操作。

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            from pyvqnet.utils import set_random_seed
            import numpy as np
            set_random_seed(42)
            a = np.arange(6).reshape(2, 3).astype(np.float32)
            t = QTensor(a)
            value = 42

            t.fill_rand_signed_uniform_(value)
            print(t)

            # [[ 29.3718  , 27.51417 , 15.379172],
            #  [ 41.62058 ,-25.133585,-28.762613]]
            # <QTensor [2, 3] DEV_CPU float32>

    .. py:method:: fill_rand_uniform_(v=1)

        用从均匀分布中随机采样的值填充 QTensor 。用缩放因子 v 对生成的随机采样的值进行缩放。

        :param v: 缩放因子, 默认1。

        :return: 无。

        .. note::

            该操作为原地操作。如果张量的 ``requires_grad`` 为 ``True``,自动微分不会记录这次修改,并发出 ``UserWarning``;此时请改用非原地的操作。

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            from pyvqnet.utils import set_random_seed
            import numpy as np
            set_random_seed(42)
            a = np.arange(6).reshape(2, 3).astype(np.float32)
            t = QTensor(a)
            value = 42
            t.fill_rand_uniform_(value)
            print(t)

            # [[35.685898 ,34.757084 ,28.689587 ],
            #  [41.81029  , 8.4332075, 6.618693 ]]
            # <QTensor [2, 3] DEV_CPU float32>

    .. py:method:: fill_rand_normal_(m=0, s=1, fast_math=True)

        生成均值为 m 和方差 s 产生正态分布元素, 并填充到张量中。

        :param m: 均值, 默认0。
        :param s: 方差, 默认1。
        :param fast_math: 是否使用快速方法产生高斯分布, 默认True。

        :return: 无。

        .. note::

            该操作为原地操作。如果张量的 ``requires_grad`` 为 ``True``,自动微分不会记录这次修改,并发出 ``UserWarning``;此时请改用非原地的操作。

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            from pyvqnet.utils import set_random_seed
            import numpy as np
            set_random_seed(42)
            a = np.arange(6).reshape(2, 3).astype(np.float32)
            t = QTensor(a)
            t.fill_rand_normal_(2, 10, True)
            print(t)

            # [[ 4.8241215, 5.0147934,-2.63997  ],
            #  [-2.054537 , 4.737082 ,-0.3090385]]
            # <QTensor [2, 3] DEV_CPU float32>

    .. py:method:: transpose(new_dims=None)

        反转张量的轴。如果 new_dims = None, 则反转所有轴。

        :param new_dims: 列表形式储存的新的轴顺序。

        :return:  新的 QTensor 。

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            import numpy as np
            R, C = 3, 4
            a = np.arange(R * C).reshape([2, 2, 3]).astype(np.float32)
            t = QTensor(a)
            rlt = t.transpose([2,0,1])
            print(rlt)
            # [
            # [[0., 3.],
            #  [6., 9.]],
            # [[1., 4.],
            #  [7., 10.]],
            # [[2., 5.],
            #  [8., 11.]]
            # ]
 


    .. py:method:: reshape(shape)

        改变 QTensor 的形状, 返回一个新的张量。

        :param shape: 新的形状。

        :return: 新形状的 QTensor 。

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            import numpy as np
            R, C = 3, 4
            a = np.arange(R * C).reshape(R, C).astype(np.float32)
            t = QTensor(a)
            reshape_t = t.reshape([C, R])
            print(reshape_t)
            # [
            # [0., 1., 2.],
            # [3., 4., 5.],
            # [6., 7., 8.],
            # [9., 10., 11.]
            # ]


    .. py:method:: __getitem__()

        支持对 QTensor 使用 切片索引, 下标, 或使用 QTensor 作为高级索引访问输入。该操作返回一个新的 QTensor 。

        通过冒号 ``:``  分隔切片参数 start:stop:step 来进行切片操作, 其中 start、stop、step 均可缺省。

        针对1-D QTensor , 则仅有单个轴上的索引或切片。

        针对2-D及以上的 QTensor , 则会有多个轴上的索引或切片。

        使用 QTensor 作为 索引, 则进行高级索引, 请参考numpy中 `高级索引 <https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html>`_ 部分。

        若作为索引的 QTensor 为逻辑运算的结果, 则进行 布尔数组索引。

        .. note:: 
            
            a[3][4][1] 形式的索引暂不支持, 使用 a[3,4,1] 形式代替。


        :param item: 以 pyslice , 整数, QTensor 构成切片索引。

        :return: 新的 QTensor。

        Example::

            from pyvqnet.tensor import tensor, QTensor
            aaa = tensor.arange(1, 61)
            aaa = aaa.reshape([4, 5, 3])
            print(aaa[0:2, 3, :2])
            # [
            # [10., 11.],
            #  [25., 26.]
            # ]
            print(aaa[3, 4, 1])
            #[59.]
            print(aaa[:, 2, :])
            # [
            # [7., 8., 9.],    
            #  [22., 23., 24.],
            #  [37., 38., 39.],
            #  [52., 53., 54.] 
            # ]
            print(aaa[2])
            # [
            # [31., 32., 33.], 
            #  [34., 35., 36.],
            #  [37., 38., 39.],
            #  [40., 41., 42.],
            #  [43., 44., 45.]
            # ]
            print(aaa[0:2, ::3, 2:])
            # [
            # [[3.],
            #  [12.]],
            # [[18.],
            #  [27.]]
            # ]
            a = tensor.ones([2, 2])
            b = QTensor([[1, 1], [0, 1]])
            b = b > 0
            c = a[b]
            print(c)
            #[1., 1., 1.]
            tt = tensor.arange(1, 56 * 2 * 4 * 4 + 1).reshape([2, 8, 4, 7, 4])
            tt.requires_grad = True
            index_sample1 = tensor.arange(0, 3).reshape([3, 1])
            index_sample2 = QTensor([0, 1, 0, 2, 3, 2, 2, 3, 3]).reshape([3, 3])
            gg = tt[:, index_sample1, 3:, index_sample2, 2:]
            print(gg)
            # [
            # [[[[87., 88.]],
            # [[983., 984.]]],
            # [[[91., 92.]],
            # [[987., 988.]]],
            # [[[87., 88.]],
            # [[983., 984.]]]],
            # [[[[207., 208.]],
            # [[1103., 1104.]]],
            # [[[211., 212.]],
            # [[1107., 1108.]]],
            # [[[207., 208.]],
            # [[1103., 1104.]]]],
            # [[[[319., 320.]],
            # [[1215., 1216.]]],
            # [[[323., 324.]],
            # [[1219., 1220.]]],
            # [[[323., 324.]],
            # [[1219., 1220.]]]]
            # ]

 

    .. py:method:: __setitem__()

        支持对 QTensor 使用 切片索引, 下标, 或使用 QTensor 作为高级索引修改输入。该操作对输入原地进行修改 。

        通过冒号 ``:``  分隔切片参数 start:stop:step 来进行切片操作, 其中 start、stop、step 均可缺省。

        针对1-D QTensor, 则仅有单个轴上的索引或切片。

        针对2-D及以上的 QTensor , 则会有多个轴上的索引或切片。

        使用 QTensor 作为 索引, 则进行高级索引, 请参考numpy中 `高级索引 <https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html>`_ 部分。

        若作为索引的 QTensor 为逻辑运算的结果, 则进行 布尔数组索引。

        .. note:: 
            
            a[3][4][1] 形式的索引暂不支持, 使用 a[3,4,1] 形式代替。


        :param item: 以 pyslice , 整数, QTensor 构成切片索引。

        :return: 无。

        Example::

            from pyvqnet.tensor import tensor
            aaa = tensor.arange(1, 61)
            aaa = aaa.reshape([4, 5, 3])
            vqnet_a2 = aaa[3, 4, 1]
            aaa[3, 4, 1] = tensor.arange(10001,
                                            10001 + vqnet_a2.size).reshape(vqnet_a2.shape)
            print(aaa)
            # [
            # [[1., 2., 3.],    
            #  [4., 5., 6.],    
            #  [7., 8., 9.],    
            #  [10., 11., 12.], 
            #  [13., 14., 15.]],
            # [[16., 17., 18.], 
            #  [19., 20., 21.], 
            #  [22., 23., 24.], 
            #  [25., 26., 27.], 
            #  [28., 29., 30.]],
            # [[31., 32., 33.], 
            #  [34., 35., 36.],
            #  [37., 38., 39.],
            #  [40., 41., 42.],
            #  [43., 44., 45.]],
            # [[46., 47., 48.],
            #  [49., 50., 51.],
            #  [52., 53., 54.],
            #  [55., 56., 57.],
            #  [58., 10001., 60.]]
            # ]
            aaa = tensor.arange(1, 61)
            aaa = aaa.reshape([4, 5, 3])
            vqnet_a3 = aaa[:, 2, :]
            aaa[:, 2, :] = tensor.arange(10001,
                                            10001 + vqnet_a3.size).reshape(vqnet_a3.shape)
            print(aaa)
            # [
            # [[1., 2., 3.],
            #  [4., 5., 6.],
            #  [10001., 10002., 10003.],
            #  [10., 11., 12.],
            #  [13., 14., 15.]],
            # [[16., 17., 18.],
            #  [19., 20., 21.],
            #  [10004., 10005., 10006.],
            #  [25., 26., 27.],
            #  [28., 29., 30.]],
            # [[31., 32., 33.],
            #  [34., 35., 36.],
            #  [10007., 10008., 10009.],
            #  [40., 41., 42.],
            #  [43., 44., 45.]],
            # [[46., 47., 48.],
            #  [49., 50., 51.],
            #  [10010., 10011., 10012.],
            #  [55., 56., 57.],
            #  [58., 59., 60.]]
            # ]
            aaa = tensor.arange(1, 61)
            aaa = aaa.reshape([4, 5, 3])
            vqnet_a4 = aaa[2, :]
            aaa[2, :] = tensor.arange(10001,
                                        10001 + vqnet_a4.size).reshape(vqnet_a4.shape)
            print(aaa)
            # [
            # [[1., 2., 3.],
            #  [4., 5., 6.],
            #  [7., 8., 9.],
            #  [10., 11., 12.],
            #  [13., 14., 15.]],
            # [[16., 17., 18.],
            #  [19., 20., 21.],
            #  [22., 23., 24.],
            #  [25., 26., 27.],
            #  [28., 29., 30.]],
            # [[10001., 10002., 10003.],
            #  [10004., 10005., 10006.],
            #  [10007., 10008., 10009.],
            #  [10010., 10011., 10012.],
            #  [10013., 10014., 10015.]],
            # [[46., 47., 48.],
            #  [49., 50., 51.],
            #  [52., 53., 54.],
            #  [55., 56., 57.],
            #  [58., 59., 60.]]
            # ]
            aaa = tensor.arange(1, 61)
            aaa = aaa.reshape([4, 5, 3])
            vqnet_a5 = aaa[0:2, ::2, 1:2]
            aaa[0:2, ::2,
                1:2] = tensor.arange(10001,
                                        10001 + vqnet_a5.size).reshape(vqnet_a5.shape)
            print(aaa)
            # [
            # [[1., 10001., 3.],
            #  [4., 5., 6.],
            #  [7., 10002., 9.],
            #  [10., 11., 12.],
            #  [13., 10003., 15.]],
            # [[16., 10004., 18.],
            #  [19., 20., 21.],
            #  [22., 10005., 24.],
            #  [25., 26., 27.],
            #  [28., 10006., 30.]],
            # [[31., 32., 33.],
            #  [34., 35., 36.],
            #  [37., 38., 39.],
            #  [40., 41., 42.],
            #  [43., 44., 45.]],
            # [[46., 47., 48.],
            #  [49., 50., 51.],
            #  [52., 53., 54.],
            #  [55., 56., 57.],
            #  [58., 59., 60.]]
            # ]
            a = tensor.ones([2, 2])
            b = tensor.QTensor([[1, 1], [0, 1]])
            b = b > 0
            x = tensor.QTensor([1001, 2001, 3001])

            a[b] = x
            print(a)
            # [
            # [1001., 2001.],
            #  [1., 3001.]
            # ]



    .. py:method:: GPU(device: int = DEV_GPU_0)

        克隆QTensor到指定的GPU设备

        device 指定存储其内部数据的设备。 当device >= DEV_GPU_0时, 数据存储在GPU上。 
        如果您的计算机有多个 GPU, 您可以指定不同的设备来存储数据。 例如, device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... 表示存储在具有不同序列号的GPU上。

        .. note::

            QTensor在不同GPU上无法进行计算。
            如果您尝试在 ID 超过验证 GPU 最大数量的 GPU 上创建 QTensor, 将引发 Cuda 错误。

        :param device: 当前保存QTensor的设备, 默认:DEV_GPU_0, device = pyvqnet.DEV_GPU_0, 存储在第一个 GPU 中, device = DEV_GPU_1, 存储在第二个 GPU 中, 依此类推。

        :return: QTensor 克隆到 GPU 设备。

        Example::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            b = a.GPU()
            print(b.device)
            #1000
 

    .. py:method:: CPU()

        克隆QTensor到特定的CPU设备

        :return: QTensor 克隆到 CPU 设备。

        Example::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            b = a.CPU()
            print(b.device)
            # 0
 

    .. py:method:: toGPU(device: int = DEV_GPU_0)

        移动QTensor到指定的GPU设备

        device 指定存储其内部数据的设备。 当device >= DEV_GPU时, 数据存储在GPU上。
        如果您的计算机有多个 GPU, 您可以指定不同的设备来存储数据。 
        例如, device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... 表示存储在具有不同序列号的GPU上。

        .. note::

            QTensor在不同GPU上无法进行计算。
            如果您尝试在 ID 超过验证 GPU 最大数量的 GPU 上创建 QTensor, 将引发 Cuda 错误。

        :param device: 当前保存QTensor的设备, 默认:DEV_GPU_0。device = pyvqnet.DEV_GPU_0, 存储在第一个 GPU 中, device = DEV_GPU_1, 存储在第二个 GPU 中, 依此类推。
        :return: QTensor 移动到 GPU 设备。

        Example::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            a = a.toGPU()
            print(a.device)
            #1000

 

    .. py:method:: toCPU()

        移动QTensor到CPU

        :return: QTensor 移动到 CPU 设备。

        Example::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            b = a.toCPU()
            print(b.device)
            # 0


    
    .. py:method:: isGPU()

        该 QTensor 的数据是否存储在 GPU 主机内存上。

        :return: 该 QTensor 的数据是否存储在 GPU 主机内存上。

        Example::
        
            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            a = a.isGPU()
            print(a)
            # False

    
    .. py:method:: isCPU()

        该 QTensor 的数据是否存储在 CPU 主机内存上。

        :return: 该 QTensor 的数据是否存储在 CPU 主机内存上。

        Example::
        
            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            a = a.isCPU()
            print(a)
            # True

    .. py:method:: astype(dtype)

        将当前QTensor转化为对应数据类型dtype, 如果dtype相同则返回自身。

        在浮点类型与复数类型之间转换时(包括 ``kfloat16`` 与 ``kcomplex32``),结果保留自动微分的梯度边,反向传播可以穿过该转换。
        复数类型转为浮点类型时保留实部。转为 ``kbool`` 或整数类型时,结果不再跟踪梯度。

        :param dtype: 目标数据类型dtype。

        :return:  QTensor。

        Example::

            from pyvqnet import kbool, kcomplex128, kcomplex32, kfloat16
            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            a = a.astype(kcomplex128)
            print(a)

            b = QTensor([0.5, -1.5], requires_grad=True).astype(kfloat16)
            print(b.dtype, b.requires_grad)

            print(b.astype(kbool).dtype, b.astype(kbool).requires_grad)

            # [2.+0.j]
            # <QTensor [1] DEV_CPU complex128>
            # float16 True
            # bool False

    .. py:method:: requires_grad_(mode=True)

        原地设置该张量是否跟踪梯度,返回自身,可以链式调用。

        只有叶子张量(leaf)可以改变该标志;对非叶子张量改变 ``requires_grad`` 会抛出 ``RuntimeError``。

        :param mode: 是否跟踪梯度,默认为 ``True``。

        :return: 当前 QTensor。

        Example::

            from pyvqnet.tensor import QTensor, tensor

            x = QTensor([2.0, 3.0])
            print(x.requires_grad_(True) is x, x.requires_grad)
            y = tensor.sums(x * x)
            y.backward()
            print(x.grad)
            x.requires_grad_(False)
            print(x.requires_grad)

            try:
                y.requires_grad_(False)
            except RuntimeError as e:
                print("RuntimeError:", e)

            # True True

            # [4.,6.]
            # <QTensor [2] DEV_CPU float32>
            # False
            # RuntimeError: you can only change requires_grad flags of leaf variables

    .. py:method:: bernoulli_(p=0.5, generator=None)

        用伯努利采样的 0/1 值原地填充张量:每个元素以概率 ``p`` 取 1,否则取 0。

        :param p: 元素取 1 的概率,范围 [0, 1],默认为 0.5。
        :param generator: 可选的 ``pyvqnet.random.Generator``;默认 ``None``,使用默认随机数引擎。

        :return: 当前 QTensor,可以链式调用。

        .. note::

            ``p`` 超出 [0, 1] 时抛出 ``ValueError``。该操作只支持浮点类型的张量,对整数或复数张量抛出 ``NotImplementedError``。
            该操作为原地操作:如果张量的 ``requires_grad`` 为 ``True``,自动微分不会记录这次修改,并发出 ``UserWarning``。

        Example::

            from pyvqnet import kint32, kfloat16
            from pyvqnet.random import Generator
            from pyvqnet.tensor import tensor
            from pyvqnet.utils import set_random_seed

            set_random_seed(42)
            t = tensor.zeros([2, 3])
            t.bernoulli_(0.7)
            print(t)
            print(t.bernoulli_(0.7) is t, t.dtype)

            g = Generator(7)
            print(tensor.zeros([2, 3]).bernoulli_(0.5, generator=g))

            print(tensor.zeros([2], dtype=kfloat16).bernoulli_(0.5).dtype)
            try:
                tensor.zeros([2]).bernoulli_(1.5)
            except ValueError as e:
                print("ValueError:", e)
            try:
                tensor.zeros([2], dtype=kint32).bernoulli_(0.5)
            except NotImplementedError as e:
                print("NotImplementedError:", e)

            # [[1.,1.,1.],
            #  [1.,0.,0.]]
            # <QTensor [2, 3] DEV_CPU float32>
            # True float32

            # [[0.,1.,1.],
            #  [0.,1.,1.]]
            # <QTensor [2, 3] DEV_CPU float32>
            # float16
            # ValueError: bernoulli_ expects a probability within [0, 1], got 1.5.
            # NotImplementedError: bernoulli_ is not implemented for 'int32'

    .. py:method:: bool()

        将张量转换为 ``pyvqnet.kbool`` 类型,等价于 ``astype(kbool)``:非零元素为 ``True``,零元素为 ``False``。

        :return: 数据类型为 ``pyvqnet.kbool`` 的 QTensor。

        Example::

            from pyvqnet import kfloat16
            from pyvqnet.tensor import QTensor

            print(QTensor([1.0, 0.0, 2.5]).bool())
            print(QTensor([1.0, 0.0], dtype=kfloat16).bool().dtype)
            print(QTensor([True, False]).bool().dtype)

            # [ True,False, True]
            # <QTensor [3] DEV_CPU bool>
            # bool
            # bool

半精度数据类型:kfloat16 与 kcomplex32
*********************************************

``pyvqnet.kfloat16`` 与 ``pyvqnet.kcomplex32`` 是 QTensor 的两种半精度数据类型。
``pyvqnet`` 包顶层同时提供简写 ``pyvqnet.float16`` 和 ``pyvqnet.complex32``,它们与 ``kfloat16`` 、 ``kcomplex32`` 是同一个对象。

==========================================  ==========  ===================================================
dtype                                       bytes       description
==========================================  ==========  ===================================================
``pyvqnet.kfloat16``                        2           16比特半精度浮点数
``pyvqnet.kcomplex32``                      4           32比特复数,由两个 ``float16`` 构成
==========================================  ==========  ===================================================

- ``kfloat16`` 是普通的浮点数据类型:创建函数的 ``dtype`` 参数、``QTensor`` 的构造函数和 ``astype`` 都接受它。
  传入 ``numpy.float16`` 数组且不指定 ``dtype`` 时,张量保持 ``kfloat16``; ``to_numpy()`` 与 ``numpy()`` 返回 ``numpy.float16`` 数组。
  ``pyvqnet.dtype.set_default_dtype`` 也可以把它设为默认数据类型。
- ``kcomplex32`` 采用存储与计算分离的方式:数据以两个 ``float16`` 存储(每个复数占 4 字节),
  算子内部的运算在 ``complex64`` / ``float32`` 中完成,结果再舍入回 ``complex32``。
  GPU 上有一部分算子直接在 complex32 存储上使用专用的 CUDA 内核;其余算子以及 CPU 上的算子在计算时把操作数升为 ``complex64``,算完再把结果舍入回 ``complex32``。
- ``pyvqnet.qnn.vqc.QMachine(num_wires, dtype=kcomplex32)`` 的态矢 ``states`` 以 complex32 存储;
  量子门的角度以 float16 存储,门矩阵为 complex32,门内部的运算保持 complex64 / float32。
  测量结果(例如 ``MeasureAll`` 、 ``Probability`` )为 ``kfloat16``;输入和含参门参数的梯度仍为 ``kfloat32``。

.. note::

    ``float16`` 只有约 3 位十进制有效数字,绝对值超过 65504 的值会变成 ``inf``。
    例如 ``1/3`` 存为 ``0.333252``, ``1e5`` 存为 ``inf``。对精度敏感的计算请使用 ``kfloat32`` / ``kcomplex64``。

类型提升与转换规则
=============================

- ``complex32`` 与 ``complex32`` 、 ``float16`` 或 Python 的 int / float / complex 标量运算,结果仍为 ``complex32``;
  与 ``complex64`` 或 ``float32`` 张量运算时,结果提升为 ``complex64``。``float16`` 张量与 Python 复数标量相加得到 ``complex32``。
- 结果为实数的算子返回 ``kfloat16``: ``abs`` 、 ``real`` 、 ``imag`` 、 ``norm`` 、 ``frobenius_norm`` 、 ``std`` 、 ``var``。
  结果为真值的算子返回 ``kbool``: ``equal`` 、 ``not_equal`` 、 ``isfinite`` 、 ``isinf`` 、 ``isnan`` 。
- ``astype`` 在浮点与复数类型之间转换时保留自动微分的梯度边;复数转实数类型时保留实部;
  转为 ``kbool`` 或整数类型时,结果不再跟踪梯度。
- ``to_numpy()``: numpy 没有 complex32 类型,complex32 张量会被转换为 ``complex64`` 数组;
  零拷贝的 ``numpy()`` 对 complex32 张量抛出 ``RuntimeError``。

Example::

        import numpy as np

        import pyvqnet
        from pyvqnet import (
            kbool,
            kcomplex32,
            kcomplex64,
            kfloat16,
            kfloat32,
        )
        from pyvqnet.tensor import QTensor, tensor

        a = QTensor([1 + 2j, 0.5 - 1j, -0.25 + 0.75j], dtype=kcomplex32)
        print(a.dtype, a.element_size())
        print(a)

        f = QTensor([1.5, 2.5, -0.125], dtype=kfloat16)
        print(f.dtype, f.element_size())
        print(f)

        print(tensor.zeros([2, 2], dtype=kcomplex32).dtype)
        print(tensor.eye(2, dtype=kcomplex32))
        print(tensor.ones([2], dtype=kfloat16).dtype)
        print(tensor.randn([2], dtype=kfloat16).dtype)

        print(QTensor(np.array([1.0, -2.5], dtype=np.float16)).dtype)

        print(pyvqnet.complex32 is kcomplex32, pyvqnet.float16 is kfloat16)

        c64 = QTensor([1 / 3 + 0j], dtype=kcomplex64)
        c32 = c64.astype(kcomplex32)
        print(c32.dtype, c32)
        print(c32.astype(kcomplex64).dtype, c32.astype(kcomplex64))
        print(f.astype(kcomplex32).dtype)
        print(a.astype(kfloat16))
        print(a.astype(kbool))

        print(a.to_numpy().dtype)          # numpy has no complex32: to_numpy() upcasts
        print(f.to_numpy().dtype)          # float16 is a numpy dtype
        try:
            a.numpy()                      # zero-copy view is impossible for complex32
        except RuntimeError as e:
            print("RuntimeError:", e)

        w = QTensor([1 + 1j, 2 - 1j, 0.5 + 0j], dtype=kcomplex32)
        h = QTensor([1.0, 2.0, 0.5], dtype=kfloat16)
        print((a + w).dtype)               # complex32 + complex32
        print((a + a.astype(kcomplex64)).dtype)   # complex32 + complex64
        print((a * h).dtype)               # complex32 + float16
        print((a + h.astype(kfloat32)).dtype)     # complex32 + float32
        print((a + 1).dtype, (a * 1.5).dtype)     # python int / float scalar
        print((a + (1 + 0j)).dtype)               # python complex scalar
        print((h + (1 + 0j)).dtype)               # float16 + python complex scalar

        print(tensor.abs(a))
        print(tensor.real(a).dtype, tensor.imag(a).dtype)
        print(tensor.sums(a))
        print(tensor.std(a))

        # complex32 4

        # [ 1.  +2.j  , 0.5 -1.j  ,-0.25+0.75j]
        # <QTensor [3] DEV_CPU complex32>
        # float16 2

        # [ 1.5  , 2.5  ,-0.125]
        # <QTensor [3] DEV_CPU float16>
        # complex32

        # [[1.+0.j,0.+0.j],
        #  [0.+0.j,1.+0.j]]
        # <QTensor [2, 2] DEV_CPU complex32>
        # float16
        # float16
        # float16
        # True True
        # complex32 
        # [0.333252+0.j]
        # <QTensor [1] DEV_CPU complex32>
        # complex64 
        # [0.333252+0.j]
        # <QTensor [1] DEV_CPU complex64>
        # complex32

        # [ 1.  , 0.5 ,-0.25]
        # <QTensor [3] DEV_CPU float16>

        # [ True, True, True]
        # <QTensor [3] DEV_CPU bool>
        # complex64
        # float16
        # RuntimeError: complex32 has no numpy zero-copy format; use to_numpy() (upcasts to complex64).
        # complex32
        # complex64
        # complex32
        # complex64
        # complex32 complex32
        # complex32
        # complex32

        # [2.236 ,1.118 ,0.7905]
        # <QTensor [3] DEV_CPU float16>
        # float16 float16

        # 1.25+1.75j
        # <QTensor [] DEV_CPU complex32>

        # 1.633
        # <QTensor [] DEV_CPU float16>

complex32 张量的算子支持
===============================

下面的列表针对 native 后端(``pyvqnet-ad``),在 CPU 与 GPU 上均经过验证。

- **支持,结果为 complex32**:

  - 创建: ``QTensor(data, dtype=kcomplex32)`` 、 ``ones`` 、 ``zeros`` 、 ``full`` 、 ``empty`` 、 ``ones_like`` 、 ``zeros_like`` 、 ``full_like`` 、 ``eye`` 、 ``randu`` 、 ``randn`` 。
  - 逐元素: ``add`` 、 ``sub`` 、 ``mul`` 、 ``divide`` (含 ``+`` 、 ``-`` 、 ``*`` 、 ``/`` 运算符) 、 ``neg`` 、 ``exp`` 、 ``log`` 、 ``sqrt`` 、 ``square`` 、 ``power`` 、 ``sin`` 、 ``cos`` 、 ``tan`` 、 ``tanh`` 、 ``sinh`` 、 ``cosh`` 、 ``acos`` 、 ``asin`` 、 ``atan`` 、 ``sigmoid`` 、 ``conj`` 、 ``adjoint`` 。
  - 归约与线性代数: ``sums`` 、 ``mean`` 、 ``trace`` 、 ``matmul`` 、 ``kron`` 、 ``einsum`` 。
  - 选择与形状: ``where`` 、 ``broadcast`` 、 ``broadcast_to`` 、 ``select`` 、 ``concatenate`` 、 ``stack`` 、 ``permute`` 、 ``transpose`` 、 ``tile`` 、 ``squeeze`` 、 ``unsqueeze`` 、 ``moveaxis`` 、 ``swapaxis`` 、 ``masked_fill`` 、 ``flatten`` 、 ``reshape`` 、 ``flip`` 、 ``gather`` 、 ``scatter`` 、 ``diagonal`` 、 ``triu`` 、 ``tril`` 、 ``pad`` 。
  - 索引与拷贝: 切片、整数、 ``QTensor`` 索引和布尔掩码的读取;切片与整数位置的赋值( ``__setitem__`` ); ``astype`` 、 ``to_numpy`` 、 ``toGPU`` 、 ``toCPU`` 。
  - 自动微分: 上述算子的梯度也是 complex32。

- **GPU 上直接使用 complex32 CUDA 内核的算子**: ``add`` 、 ``sub`` 、 ``mul`` 、 ``divide`` 、 ``sin`` 、 ``cos`` 、 ``tan`` 、 ``acos`` 、 ``asin`` 、 ``atan`` 、 ``sinh`` 、 ``cosh`` 、 ``tanh`` 、 ``exp`` 、 ``log`` 、 ``sqrt`` 、 ``square`` 、 ``neg`` 、 ``sums`` 、 ``abs`` 、 ``cat`` 、 ``concatenate`` 、 ``stack`` 、 ``contiguous`` 、 ``flip`` 、 ``roll`` 、 ``tile`` 、 ``diag`` 、 ``tril`` 、 ``triu`` 、 ``gather`` 、 ``scatter`` 、 ``trace`` 、 ``equal`` 、 ``not_equal`` 、 ``isfinite`` 、 ``isinf`` 、 ``isnan`` 、 ``nonzero`` 、 ``pow`` 、 ``power`` 、 ``rdivide`` 、 ``std`` 、 ``var`` 、 ``kron`` 、 ``outer`` 。其余受支持的算子在 GPU 上同样先升为 ``complex64`` 再计算。

- **不支持复数输入,complex32 与 complex64 相同**: ``arange`` 、 ``linspace`` 、 ``logspace`` (``dtype`` 为复数时) 、 ``softmax`` 、 ``log_softmax`` 、 ``clip`` 、 ``maximum`` 、 ``minimum`` 、 ``mod`` 、 ``floor`` 、 ``ceil`` 、 ``round`` 、 ``relu`` 、 ``swiglu`` 、 ``sort`` 、 ``argsort`` 、 ``topK`` 、 ``argtopK`` 、 ``max`` 、 ``min`` 、 ``argmax`` 、 ``argmin`` 、 ``median`` 、 ``greater`` 、 ``less`` 、 ``greater_equal`` 、 ``less_equal`` (含 ``>`` 、 ``<`` 等运算符) 、 ``isneginf`` 、 ``isposinf`` 、 ``binomial`` 、 ``multinomial`` 、 ``bernoulli_`` 。

- **complex32 不支持而 complex64 支持的算子**:

  - CPU 与 GPU: ``cumsum`` 、 ``eigh`` 、 ``eigvalsh`` 、 ``logical_or`` 、 ``logical_xor`` 、 ``fill_rand_binary_`` 、 ``fill_`` 传入 Python 复数标量(实数标量可以)、零拷贝的 ``numpy()`` 。
  - 仅 CPU: ``reciprocal`` 、 ``sign`` 。
  - 仅 GPU: ``QTensor.all()`` 、 ``QTensor.any()`` 、 ``logical_not`` 、 ``__setitem__`` 的 ``QTensor`` 索引与布尔掩码赋值。

  不支持的算子在调用时抛出异常。

Example::

        import numpy as np

        from pyvqnet import kcomplex32, kcomplex64
        from pyvqnet.tensor import QTensor, tensor

        a = QTensor(np.array([1 + 2j, 0.5 - 1j, -0.25 + 0.75j, 2 + 0j]), dtype=kcomplex32)
        m = QTensor(np.array([[1 + 1j, 2 - 1j], [0.5 + 0j, -1 + 2j]]), dtype=kcomplex32)

        print((a + a).dtype, (a * a).dtype, (a / a).dtype, (-a).dtype)
        print(a * a)
        print(tensor.exp(a).dtype, tensor.sqrt(a).dtype, tensor.sin(a).dtype)
        print(tensor.conj(a))

        print(tensor.sums(a))
        print(tensor.mean(a).dtype, tensor.trace(m).dtype)
        print(tensor.matmul(m, m))
        print(tensor.kron(m, m).shape, tensor.einsum("ij,jk->ik", m, m).dtype)

        print((a == a).dtype, tensor.equal(a, a).dtype, tensor.isfinite(a).dtype)
        mask = QTensor(np.array([True, False, True, False]))
        print(tensor.where(mask, a, a).dtype)
        print(a[mask].dtype)

        print(tensor.transpose(m, [1, 0]).dtype, tensor.reshape(m, [4]).dtype)
        print(tensor.concatenate([m, m], 0).dtype, tensor.stack([m, m]).dtype)
        m[0, 0] = QTensor(np.array([9 + 9j]), dtype=kcomplex32)
        print(m)
        m += m
        print(m.dtype)
        m.mul_(m.astype(kcomplex64))   # in-place ops narrow a complex64 result back into the complex32 tensor
        print(m.dtype)
        print(m)

        for name, call in [
            ("tensor.cumsum(a, 0)", lambda: tensor.cumsum(a, 0)),
            ("tensor.reciprocal(a)", lambda: tensor.reciprocal(a)),
            ("tensor.sign(a)", lambda: tensor.sign(a)),
            ("tensor.eigvalsh(m)", lambda: tensor.eigvalsh(m)),
            ("tensor.logical_or(a, a)", lambda: tensor.logical_or(a, a)),
            ("a.fill_(1 + 1j)", lambda: a.fill_(1 + 1j)),
        ]:
            try:
                call()
                print(name, "-> ok")
            except RuntimeError as e:
                print(name, "->", type(e).__name__, ":", e)

        print(tensor.cumsum(a.astype(kcomplex64), 0).dtype)

        # complex32 complex32 complex32 complex32

        # [-3.  +4.j   ,-0.75-1.j   ,-0.5 -0.375j, 4.  +0.j   ]
        # <QTensor [4] DEV_CPU complex32>
        # complex32 complex32 complex32

        # [ 1.  -2.j  , 0.5 +1.j  ,-0.25-0.75j, 2.  -0.j  ]
        # <QTensor [4] DEV_CPU complex32>

        # 3.25+1.75j
        # <QTensor [] DEV_CPU complex32>
        # complex32 complex32

        # [[ 1.+1.5j, 3.+6.j ],
        #  [ 0.+1.5j,-2.-4.5j]]
        # <QTensor [2, 2] DEV_CPU complex32>
        # [4, 4] complex32
        # bool bool bool
        # complex32
        # complex32
        # complex32 complex32
        # complex32 complex32

        # [[ 9. +9.j, 2. -1.j],
        #  [ 0.5+0.j,-1. +2.j]]
        # <QTensor [2, 2] DEV_CPU complex32>
        # complex32
        # complex32

        # [[  0.+648.j, 12. -16.j],
        #  [  1.  +0.j,-12. -16.j]]
        # <QTensor [2, 2] DEV_CPU complex32>
        # tensor.cumsum(a, 0) -> RuntimeError : cumsum is not implemented for complex32 storage on CPU.
        # tensor.reciprocal(a) -> RuntimeError : Not impl [cpu_inv_impl_native] for dtype: VQNET_DTYPE_COMPLEX32_T.
        # tensor.sign(a) -> RuntimeError : Not impl [cpu_sign_impl_native] for dtype: VQNET_DTYPE_COMPLEX32_T.
        # tensor.eigvalsh(m) -> RuntimeError : Not supported data type for eigvalsh_impl(), got VQNET_DTYPE_COMPLEX32_T
        # tensor.logical_or(a, a) -> RuntimeError : this function not supports this dtype
        # a.fill_(1 + 1j) -> RuntimeError : Fill complex value with non-complex tensor.
        # complex64

在 GPU 上,上表中"仅 CPU"一行的算子( ``reciprocal`` 、 ``sign`` )可用,"仅 GPU"一行的算子
( ``all()`` 、 ``any()`` 、 ``logical_not`` 以及用 ``QTensor`` 索引或布尔掩码赋值)不可用。下面的例子在单张 GPU 上运行:

Example::

        import numpy as np
        import pyvqnet
        from pyvqnet import DEV_GPU_0, kcomplex32, kfloat16
        from pyvqnet.tensor import QTensor, tensor

        a = QTensor(np.array([1 + 2j, 0.5 - 1j, -0.25 + 0.75j, 2 + 0j]), dtype=kcomplex32,
                    device=DEV_GPU_0)
        print(a.device, a.dtype)
        print((a * a).dtype, tensor.exp(a).dtype, tensor.reciprocal(a).dtype)
        print(tensor.sums(a))

        x = QTensor(np.array([0.5 + 0.5j, 1 - 0.5j]), dtype=kcomplex32, device=DEV_GPU_0,
                    requires_grad=True)
        y = tensor.sums(tensor.mul(x, x))
        y.backward(QTensor(np.array([1 + 0j]), dtype=kcomplex32, device=DEV_GPU_0))
        print(x.grad.dtype, x.grad)

        for name, call in (("a.all()", a.all), ("tensor.logical_not(a)", lambda: tensor.logical_not(a)),
                           ("a[mask] = v", lambda: a.__setitem__(
                               QTensor(np.array([True, False, True, False]), device=DEV_GPU_0),
                               QTensor(np.array([9 + 9j, 9 + 9j]), dtype=kcomplex32, device=DEV_GPU_0)))):
            try:
                call()
                print(name, "-> ok")
            except (RuntimeError, ValueError) as e:
                print(name, "->", type(e).__name__, ":", str(e).split("\n")[0])

        # cuda:0 complex32
        # complex32 complex32 complex32

        # 3.25+1.75j
        # <QTensor [] DEV_GPU_0 complex32>
        # complex32 
        # [1.-1.j,2.+1.j]
        # <QTensor [2] DEV_GPU_0 complex32>
        # a.all() -> RuntimeError : Not impl for dtype: VQNET_DTYPE_COMPLEX32_T
        # tensor.logical_not(a) -> RuntimeError : Not impl [_cuda_logical_not_impl] for dtype: VQNET_DTYPE_COMPLEX32_T.
        # a[mask] = v -> ValueError : only support slice and integar index, got: 

complex32 与 float16 张量的梯度
===============================

自动微分可以穿过 complex32 与 float16 的算子以及它们与 ``complex64`` / ``float32`` 之间的 ``astype`` 转换。

Example::

        import numpy as np

        from pyvqnet import kcomplex32, kcomplex64, kfloat16, kfloat32
        from pyvqnet.tensor import QTensor, tensor

        x = QTensor(np.array([0.5 + 0.5j, 1 - 0.5j]), dtype=kcomplex32, requires_grad=True)
        w = QTensor(np.array([[1 + 1j, 2 - 1j], [0.5 + 0j, -1 + 2j]]), dtype=kcomplex32,
                    requires_grad=True)

        y = tensor.matmul(w, x.reshape([2, 1]))        # complex32, computed in complex64
        loss = tensor.sums(y * y)
        print(y.dtype, loss.dtype)
        loss.backward(QTensor(np.array([1 + 0j]), dtype=kcomplex32))
        print(x.grad.dtype, w.grad.dtype)
        print(x.grad)
        print(w.grad)

        x64 = x.detach().astype(kcomplex64)
        x64.requires_grad = True
        w64 = w.detach().astype(kcomplex64)
        w64.requires_grad = True
        y64 = tensor.matmul(w64, x64.reshape([2, 1]))
        loss64 = tensor.sums(y64 * y64)
        loss64.backward(QTensor(np.array([1 + 0j]), dtype=kcomplex64))
        print(loss64.dtype, x64.grad.dtype)
        print(x64.grad)
        print(w64.grad)

        p = QTensor(np.array([0.3, -0.7]), dtype=kfloat32, requires_grad=True)
        q = p.astype(kfloat16)                                  # float32 -> float16
        loss = tensor.sums(q * q)
        loss.backward()
        print(loss.dtype, p.grad.dtype)
        print(p.grad)                                           # 2 * p, rounded through float16

        c = QTensor(np.array([0.5 + 0.5j]), dtype=kcomplex64, requires_grad=True)
        d = c.astype(kcomplex32)                                # complex64 -> complex32
        r = d * d
        r.backward(QTensor(np.array([1 + 0j]), dtype=kcomplex32))
        print(r.dtype, c.grad.dtype, c.grad)

        h = QTensor(np.array([[1, 2], [3, 4]], dtype=np.float16), dtype=kfloat16)
        print(tensor.matmul(h, h).dtype, tensor.softmax(h, 1).dtype)
        print(tensor.topK(h, 1, 1))

        # complex32 complex32
        # complex32 complex32

        # [ 5.25 -3.75j,-7.5 +11.5j ]
        # <QTensor [2] DEV_CPU complex32>

        # [[ 2.5 -0.5j , 2.  +3.5j ],
        #  [-2.5 -3.j  , 3.25-5.25j]]
        # <QTensor [2, 2] DEV_CPU complex32>
        # complex64 complex64

        # [ 5.25 -3.75j,-7.5 +11.5j ]
        # <QTensor [2] DEV_CPU complex64>

        # [[ 2.5 -0.5j , 2.  +3.5j ],
        #  [-2.5 -3.j  , 3.25-5.25j]]
        # <QTensor [2, 2] DEV_CPU complex64>
        # float16 float32

        # [ 0.6000977,-1.4003906]
        # <QTensor [2] DEV_CPU float32>
        # complex32 complex64 
        # [1.-1.j]
        # <QTensor [1] DEV_CPU complex64>
        # float16 float16

        # [[2.],
        #  [4.]]
        # <QTensor [2, 1] DEV_CPU float16>

complex32 态矢上的量子门、期望与梯度
=====================================

下面的例子在 complex32 的模拟器上作用 ``rx`` 、 ``CNOT`` 和一个含参的 ``RY`` 门,计算 ``Z1`` 的期望、概率和一个厄密观测量的期望,并对输入和含参门的参数求梯度。作为对照,同样的线路也在 complex64 上运行一次。

Example::

        import numpy as np

        from pyvqnet import kcomplex32, kcomplex64
        from pyvqnet.nn import Module
        from pyvqnet.qnn.vqc import MeasureAll, Probability, QInfoMeasurement, QMachine, RY, cnot, rx
        from pyvqnet.tensor import QTensor, tensor

        H = np.array([[1, 1], [1, -1]], dtype=np.complex64)

        class QM(Module):
            def __init__(self, dtype):
                super().__init__()
                self.device = QMachine(2, dtype=dtype)
                self.ry = RY(trainable=True, wires=1, init_params=tensor.QTensor([0.6]))
                self.measure = MeasureAll(obs={"Z1": 1.0})

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(x.shape[0])
                rx(q_machine=self.device, wires=0, params=x[:, [0]])
                cnot(q_machine=self.device, wires=[0, 1])
                self.ry(q_machine=self.device)
                return self.measure(q_machine=self.device)

        x32 = QTensor([[0.3], [0.7]], requires_grad=True)
        qm32 = QM(kcomplex32)
        y32 = qm32(x32)
        print(qm32.device.states.dtype)
        print(y32.dtype)
        print(y32)

        x64 = QTensor([[0.3], [0.7]], requires_grad=True)
        qm64 = QM(kcomplex64)
        y64 = qm64(x64)
        print(qm64.device.states.dtype, y64.dtype)
        print(y64)

        y32.backward()
        y64.backward()
        print(x32.grad.dtype, qm32.ry.params.dtype, qm32.ry.params.grad.dtype)
        print(x32.grad)
        print(x64.grad)
        print(qm32.ry.params.grad)
        print(qm64.ry.params.grad)

        probs = Probability(wires=[0, 1])
        print(probs(q_machine=qm32.device).dtype)
        print(probs(q_machine=qm32.device))

        herm = QInfoMeasurement("expectation", obs={"wires": (1,), "observables": tensor.to_tensor(H)})
        print(herm(q_machine=qm32.device).dtype)
        print(herm(q_machine=qm32.device))

        # complex32
        # float16

        # [[0.788 ],
        #  [0.6313]]
        # <QTensor [2, 1] DEV_CPU float16>
        # complex64 float32

        # [[0.7884732],
        #  [0.6312516]]
        # <QTensor [2, 1] DEV_CPU float32>
        # float32 float32 float32

        # [[-0.2437744],
        #  [-0.5317383]]
        # <QTensor [2, 1] DEV_CPU float32>

        # [[-0.2439034],
        #  [-0.5316957]]
        # <QTensor [2, 1] DEV_CPU float32>

        # [[-0.9716797]]
        # <QTensor [1, 1] DEV_CPU float32>

        # [[-0.971286]]
        # <QTensor [1, 1] DEV_CPU float32>
        # float16

        # [[0.8916  ,0.0854  ,0.001953,0.02037 ],
        #  [0.8057  ,0.0772  ,0.01029 ,0.10736 ]]
        # <QTensor [2, 4] DEV_CPU float16>
        # float16

        # [[1.327],
        #  [1.063]]
        # <QTensor [2, 1] DEV_CPU float16>


创建函数
*************

.. _ones:

ones
==============================

.. py:function:: pyvqnet.tensor.ones(shape,device=pyvqnet.DEV_CPU,dtype=None)

    创建元素全一的 QTensor 。

    :param shape: 数据的形状。
    :param device: 储存在哪个设备上,默认: pyvqnet.DEV_CPU,在CPU上。
    :param dtype: 参数的数据类型,默认: None,使用默认数据类型:kfloat32,代表32位浮点数。

    :return: 返回新的 QTensor 。

    Example::

        from pyvqnet.tensor import tensor

        x = tensor.ones([2, 3])
        print(x)

        # [
        # [1., 1., 1.],
        # [1., 1., 1.]
        # ]

ones_like
==============================

.. py:function:: pyvqnet.tensor.ones_like(t: pyvqnet.tensor.QTensor,device=pyvqnet.DEV_CPU,dtype=None)

    创建元素全一的 QTensor ,形状和输入的 QTensor 一样。

    :param t: 输入 QTensor 。
    :param device: 储存在哪个设备上,默认: pyvqnet.DEV_CPU,在CPU上。
    :param dtype: 参数的数据类型,默认: None,跟输入的dtype一样。

    :return: 新的全一  QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.ones_like(t)
        print(x)

        # [1., 1., 1.]


full
==============================

.. py:function:: pyvqnet.tensor.full(shape, value, device=pyvqnet.DEV_CPU, dtype=None)

    创建一个指定形状的 QTensor 并用特定值填充它。

    :param shape: 要创建的张量形状。
    :param value: 填充的值。
    :param device: 储存在哪个设备上,默认: pyvqnet.DEV_CPU,在CPU上。
    :param dtype: 参数的数据类型,默认: None,使用默认数据类型:kfloat32,代表32位浮点数。

    :return: 输出新 QTensor 。 

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        shape = [2, 3]
        value = 42
        t = tensor.full(shape, value)
        print(t)
        # [
        # [42., 42., 42.],
        # [42., 42., 42.]
        # ]


full_like
==============================

.. py:function:: pyvqnet.tensor.full_like(t, value, device=pyvqnet.DEV_CPU,dtype=None)

    创建一个形状和输入一样的 QTensor,所有元素填充 value 。

    :param t: 输入 QTensor 。
    :param value: 填充 QTensor 的值。
    :param device: 储存在哪个设备上,默认: pyvqnet.DEV_CPU,在CPU上。
    :param dtype: 参数的数据类型,默认: None,跟输入的dtype一样。

    :return: 输出 QTensor。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        a = tensor.randu([3,5])
        value = 42
        t = tensor.full_like(a, value)
        print(t)
        # [
        # [42., 42., 42., 42., 42.],    
        # [42., 42., 42., 42., 42.],    
        # [42., 42., 42., 42., 42.]     
        # ]
        

zeros
==============================

.. py:function:: pyvqnet.tensor.zeros(shape, device=pyvqnet.DEV_CPU,dtype=None)

    创建输入形状大小的全零 QTensor 。

    :param shape: 输入形状。
    :param device: 储存在哪个设备上,默认: pyvqnet.DEV_CPU,在CPU上。
    :param dtype: 参数的数据类型,默认: None,使用默认数据类型:kfloat32,代表32位浮点数。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = tensor.zeros([2, 3, 4])
        print(t)
        # [
        # [[0., 0., 0., 0.],
        #  [0., 0., 0., 0.],
        #  [0., 0., 0., 0.]],
        # [[0., 0., 0., 0.],
        #  [0., 0., 0., 0.],
        #  [0., 0., 0., 0.]]
        # ]
        

zeros_like
==============================

.. py:function:: pyvqnet.tensor.zeros_like(t: pyvqnet.tensor.QTensor, device=pyvqnet.DEV_CPU,dtype=None)

    创建一个形状和输入一样的 QTensor,所有元素为0 。

    :param t: 输入参考 QTensor 。
    :param device: 储存在哪个设备上,默认: pyvqnet.DEV_CPU,在CPU上。
    :param dtype: 参数的数据类型,默认: None,跟输入的dtype一样。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.zeros_like(t)
        print(x)

        # [0., 0., 0.]
        


arange
==============================

.. py:function:: pyvqnet.tensor.arange(start, end, step=1, device=pyvqnet.DEV_CPU,dtype=None,requires_grad=False)

    创建一个在给定间隔内具有均匀间隔值的一维 QTensor 。

    :param start: 间隔开始。
    :param end: 间隔结束。
    :param step: 值之间的间距,默认为1。
    :param device: 要使用的设备,默认 = pyvqnet.DEV_CPU,使用 CPU 设备。
    :param dtype: 参数的数据类型,默认: None,使用默认数据类型:kfloat32,代表32位浮点数。
    :param requires_grad: 是否计算梯度,默认为False。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = tensor.arange(2, 30, 4)
        print(t)

        # [ 2.,  6., 10., 14., 18., 22., 26.]
        

linspace
==============================

.. py:function:: pyvqnet.tensor.linspace(start, end, nums, device=pyvqnet.DEV_CPU,dtype=None,requires_grad= False)

    创建一维 QTensor ,其中的元素为区间 start 和 end 上均匀间隔的共 nums 个值。

    :param start: 间隔开始。
    :param end: 间隔结束。
    :param nums: 间隔的个数。
    :param device: 要使用的设备,默认 = pyvqnet.DEV_CPU ,使用 CPU 设备。
    :param dtype: 参数的数据类型,默认: None,使用默认数据类型:kfloat32,代表32位浮点数。
    :param requires_grad: 是否计算梯度,默认为False。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        start, stop, num = -2.5, 10, 10
        t = tensor.linspace(start, stop, num)
        print(t)
        #[-2.5000000, -1.1111112, 0.2777777, 1.6666665, 3.0555553, 4.4444442, 5.8333330, 7.2222219, 8.6111107, 10.]

logspace
==============================

.. py:function:: pyvqnet.tensor.logspace(start, end, nums, base, device=pyvqnet.DEV_CPU,dtype=None, requires_grad= False)

    在对数刻度上创建具有均匀间隔值的一维 QTensor。

    :param start: ``base ** start`` 是起始值
    :param end: ``base ** end`` 是序列的最终值
    :param nums: 要生成的样本数
    :param base: 对数刻度的基数
    :param device: 要使用的设备,默认 = pyvqnet.DEV_CPU ,使用 CPU 设备。
    :param dtype: 参数的数据类型,默认: None,使用默认数据类型:kfloat32,代表32位浮点数。
    :param requires_grad: 是否计算梯度,默认为False。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        start, stop, steps, base = 0.1, 1.0, 5, 10.0
        t = tensor.logspace(start, stop, steps, base)
        print(t)

        # [1.2589254, 2.1134889, 3.5481336, 5.9566211, 10.]
        

eye
==============================

.. py:function:: pyvqnet.tensor.eye(size, offset: int = 0, device=pyvqnet.DEV_CPU,dtype=None,requires_grad=False)

    创建一个 size x size 的 QTensor,对角线上为 1,其他地方为 0。

    :param size: 要创建的(正方形)QTensor 的大小。
    :param offset: 对角线的索引:0(默认)表示主对角线,正值表示上对角线,负值表示下对角线。
    :param device: 要使用的设备,默认 =pyvqnet.DEV_CPU ,使用 CPU 设备。
    :param dtype: 参数的数据类型,默认: None,使用默认数据类型:kfloat32,代表32位浮点数。
    :param requires_grad: 是否计算梯度,默认为False。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        size = 3
        t = tensor.eye(size)
        print(t)

        # [
        # [1., 0., 0.],
        # [0., 1., 0.],
        # [0., 0., 1.]
        # ]
        

diagonal
==============================
.. py:function:: pyvqnet.tensor.diagonal(t: QTensor, offset: int = 0, dim1=0, dim2=1)

    返回 :attr:`t` 的部分视图,其对角线元素相对于 :attr:`dim1` 和 :attr:`dim2` 附加为形状末尾的维度。
    :attr:`offset` 是主对角线的偏移量。

    :param t: 输入张量
    :param offset: 偏移量(0 表示主对角线,正值表示主对角线上方的第 n 条对角线,负值表示主对角线下方的第 n 条对角线)
    :param dim1: 取对角线的第一维度。默认值:0。
    :param dim2: 取对角线的第二维度。默认值:1。

    Example::

        from pyvqnet.tensor import randn,diagonal

        x = randn((2, 5, 4, 2))
        diagonal_elements = diagonal(x, offset=-1, dim1=1, dim2=2)
        print(diagonal_elements)
        # [[[-0.4641751,-0.1410288,-0.1215512, 0.5423283],
        #   [ 0.9556418, 0.0376572, 1.2571657, 0.8268463]],

        #  [[-0.7972266, 0.2080281,-0.1157126,-0.7342224],
        #   [ 1.1039937, 0.4700735, 1.0219841,-0.146358 ]]]


diag
==============================

.. py:function:: pyvqnet.tensor.diag(t, k: int = 0)

    构造对角矩阵。

    输入一个 2-D QTensor,则返回一个1D的新张量,包含
    选定对角线中的元素。
    输入一个 1-D QTensor,则返回一个2D新张量,其选定对角线元素为输入值,其余为0

    ``diag`` 可以微分:对结果求梯度时,梯度会回传到输入张量中对应的对角线元素。

    :param t: 输入 QTensor。
    :param k: 偏移量(主对角线为 0, 正数为向上偏移, 负数为向下偏移), 默认为0。

    :return: 输出 QTensor。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat32
        import numpy as np
        a = np.arange(16).reshape(4, 4).astype(np.float32)
        t = QTensor(a)
        for k in range(-3, 4):
            u = tensor.diag(t,k=k)
            print(u)

        d = QTensor([1.0, 2.0, 3.0], dtype=kfloat32, requires_grad=True)
        m = tensor.diag(d)
        tensor.sums(m).backward()
        print(d.grad)

        # [12.]
        # <QTensor [1] DEV_CPU float32>

        # [ 8.,13.]
        # <QTensor [2] DEV_CPU float32>

        # [ 4., 9.,14.]
        # <QTensor [3] DEV_CPU float32>

        # [ 0., 5.,10.,15.]
        # <QTensor [4] DEV_CPU float32>

        # [ 1., 6.,11.]
        # <QTensor [3] DEV_CPU float32>

        # [2.,7.]
        # <QTensor [2] DEV_CPU float32>

        # [3.]
        # <QTensor [1] DEV_CPU float32>

        # [1.,1.,1.]
        # <QTensor [3] DEV_CPU float32>

randu
==============================

.. py:function:: pyvqnet.tensor.randu(shape, min=0.0,max=1.0, device=pyvqnet.DEV_CPU, dtype=None, requires_grad=False,generator=None)

    创建一个具有均匀分布随机值的 QTensor 。

    :param shape: 要创建的 QTensor 的形状。
    :param min: 分布的下限,默认: 0。
    :param max: 分布的上限,默认: 1。
    :param device: 要使用的设备,默认 =pyvqnet.DEV_CPU ,使用 CPU 设备。
    :param dtype: 参数的数据类型,默认: None,使用默认数据类型:kfloat32,代表32位浮点数。
    :param requires_grad: 是否计算梯度,默认为False。
    :param generator: 可选的 ``pyvqnet.random.Generator``;默认 ``None``,使用默认随机数引擎。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        shape = [2, 3]
        t = tensor.randu(shape)
        print(t)

        # [
        # [0.0885886, 0.9570093, 0.8304565],
        # [0.6055251, 0.8721224, 0.1927866]
        # ]
        

randn
==============================

.. py:function:: pyvqnet.tensor.randn(shape, mean=0.0,std=1.0, device=pyvqnet.DEV_CPU, dtype=None, requires_grad=False,generator=None)

    创建一个具有正态分布随机值的 QTensor 。

    :param shape: 要创建的 QTensor 的形状。
    :param mean: 分布的均值,默认: 0。
    :param std: 分布的方差,默认: 1。
    :param device: 要使用的设备,默认 = pyvqnet.DEV_CPU ,使用 CPU 设备。
    :param dtype: 参数的数据类型,默认: None,使用默认数据类型:kfloat32,代表32位浮点数。
    :param requires_grad: 是否计算梯度,默认为False。
    :param generator: 可选的 ``pyvqnet.random.Generator``;默认 ``None``,使用默认随机数引擎。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        shape = [2, 3]
        t = tensor.randn(shape)
        print(t)

        # [
        # [-0.9529880, -0.4947567, -0.6399882],
        # [-0.6987777, -0.0089036, -0.5084590]
        # ]

binomial
==============================
.. py:function:: pyvqnet.tensor.binomial(total_counts, probs, generator=None)

    创建一个由 :attr:`total_count` 和 :attr:`probs` 参数化的二项分布。

    ``total_counts`` 可以是整数,也可以是与 ``probs`` 同形状的 QTensor。``probs`` 必须是浮点类型。
    对每个元素:概率小于等于 0 时采样结果为 0,概率大于等于 1 时采样结果为试验次数,试验次数小于等于 0 时采样结果为 0,概率为 ``nan`` 时结果为 ``nan``。

    :param total_counts: 伯努利试验的次数。
    :param probs: 事件概率。
    :param generator: 可选的 ``pyvqnet.random.Generator``;默认 ``None``,使用默认随机数引擎。

    :return:
        二项分布的 QTensor。

    Example::

        import pyvqnet.tensor as tensor
        from pyvqnet.random import Generator
        from pyvqnet.utils import set_random_seed

        set_random_seed(42)
        a = tensor.randu([3, 4])
        b = 1000

        c = tensor.binomial(b, a)
        print(c)

        g = Generator(7)
        print(tensor.binomial(b, a, generator=g))

        p = tensor.QTensor([0.0, 1.0, 0.5, 0.5])
        n = tensor.QTensor([1000.0, 1000.0, 1000.0, -5.0])
        print(tensor.binomial(n, p))

        # [[835.,821.,681.,992.],
        #  [238.,168.,842.,978.],
        #  [189.,877.,855.,541.]]
        # <QTensor [3, 4] DEV_CPU float32>

        # [[859.,818.,686.,998.],
        #  [214.,165.,834.,990.],
        #  [184.,880.,849.,551.]]
        # <QTensor [3, 4] DEV_CPU float32>

        # [   0.,1000., 509.,   0.]
        # <QTensor [4] DEV_CPU float32>

multinomial
==============================

.. py:function:: pyvqnet.tensor.multinomial(t, num_samples, generator=None, replacement=False)

    返回一个张量,其中每行包含 num_samples 个索引采样,来自位于张量输入的相应行中的多项式概率分布。

    :param t: 输入概率分布,仅支持浮点数,形状为 1 维或 2 维; ``kfloat16`` 与 ``kbfloat16`` 类型的输入会先转成 ``kfloat32`` 再采样。
    :param num_samples: 采样样本数,必须大于 0; ``replacement`` 为 ``False`` 时不能大于最后一维的长度。
    :param generator: 可选的 ``pyvqnet.random.Generator``;默认 ``None``,使用默认随机数引擎。
    :param replacement: 是否有放回采样,默认 ``False`` (不放回)。

    :return:
         输出采样索引,数据类型为 ``kint64``。

    Example::

        from pyvqnet import tensor
        from pyvqnet.utils import set_random_seed
        from pyvqnet.random import Generator

        set_random_seed(42)
        weights = tensor.QTensor([0.1, 10, 3, 1])
        idx = tensor.multinomial(weights, 3)
        print(idx)
        idx = tensor.multinomial(weights, 3, replacement=True)
        print(idx)
        g = Generator(7)
        print(tensor.multinomial(weights, 2, generator=g))

        for call in (
            lambda: tensor.multinomial(weights, 0),
            lambda: tensor.multinomial(weights, 5),
        ):
            try:
                call()
            except ValueError as e:
                print("ValueError:", e)

        # [1,2,3]
        # <QTensor [3] DEV_CPU int64>

        # [1,1,2]
        # <QTensor [3] DEV_CPU int64>

        # [1,0]
        # <QTensor [2] DEV_CPU int64>
        # ValueError: cannot sample n_sample <= 0 samples
        # ValueError: cannot sample n_sample > prob_dist.size(-1) samples without replacement

triu
==============================

.. py:function:: pyvqnet.tensor.triu(t, diagonal=0)

    返回输入 t 的上三角矩阵,其余部分被设为0。

    :param t: 输入 QTensor。
    :param diagonal: 偏移量(主对角线为 0, 正数为向上偏移, 负数为向下偏移), 默认:0。

    :return: 输出 QTensor。

    Example::

        from pyvqnet.tensor import tensor
        a = tensor.arange(1.0, 2 * 6 * 5 + 1.0).reshape([2, 6, 5])
        u = tensor.triu(a, 1)
        print(u)
        # [
        # [[0., 2., 3., 4., 5.],       
        #  [0., 0., 8., 9., 10.],      
        #  [0., 0., 0., 14., 15.],     
        #  [0., 0., 0., 0., 20.],      
        #  [0., 0., 0., 0., 0.],       
        #  [0., 0., 0., 0., 0.]],      
        # [[0., 32., 33., 34., 35.],   
        #  [0., 0., 38., 39., 40.],    
        #  [0., 0., 0., 44., 45.],     
        #  [0., 0., 0., 0., 50.],      
        #  [0., 0., 0., 0., 0.],       
        #  [0., 0., 0., 0., 0.]]       
        # ]

tril
==============================

.. py:function:: pyvqnet.tensor.tril(t, diagonal=0)

    返回输入 t 的下三角矩阵,其余部分被设为0。

    :param t: 输入 QTensor。
    :param diagonal: 偏移量(主对角线为 0, 正数为向上偏移, 负数为向下偏移), 默认:0。

    :return: 输出 QTensor。

    Example::

        from pyvqnet.tensor import tensor
        a = tensor.arange(1.0, 2 * 6 * 5 + 1.0).reshape([12, 5])
        u = tensor.tril(a, 1)
        print(u)
        # [
        # [1., 2., 0., 0., 0.],      
        #  [6., 7., 8., 0., 0.],     
        #  [11., 12., 13., 14., 0.], 
        #  [16., 17., 18., 19., 20.],
        #  [21., 22., 23., 24., 25.],
        #  [26., 27., 28., 29., 30.],
        #  [31., 32., 33., 34., 35.],
        #  [36., 37., 38., 39., 40.],
        #  [41., 42., 43., 44., 45.],
        #  [46., 47., 48., 49., 50.],
        #  [51., 52., 53., 54., 55.],
        #  [56., 57., 58., 59., 60.]
        # ]

数学函数
*******************************


floor
==============================

.. py:function:: pyvqnet.tensor.floor(t)

    返回一个新的 QTensor,其中元素为输入 QTensor 的向下取整。

    :param t: 输入 QTensor 。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-2.0, 2.0, 0.25)
        u = tensor.floor(t)
        print(u)

        # [-2., -2., -2., -2., -1., -1., -1., -1., 0., 0., 0., 0., 1., 1., 1., 1.]

ceil
==============================

.. py:function:: pyvqnet.tensor.ceil(t)

    返回一个新的 QTensor,其中元素为输入 QTensor 的向上取整。

    :param t: 输入 QTensor 。
    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-2.0, 2.0, 0.25)
        u = tensor.ceil(t)
        print(u)

        # [-2., -1., -1., -1., -1., -0., -0., -0., 0., 1., 1., 1., 1., 2., 2., 2.]

round
==============================

.. py:function:: pyvqnet.tensor.round(t)

    返回一个新的 QTensor,其中元素为输入 QTensor 的四舍五入到最接近的整数.

    :param t: 输入 QTensor 。
    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-2.0, 2.0, 0.4)
        u = tensor.round(t)
        print(u)

        # [-2., -2., -1., -1., -0., -0., 0., 1., 1., 2.]

sort
==============================

.. py:function:: pyvqnet.tensor.sort(t, dim: int, descending=False, stable=True)

    按指定轴对输入 QTensor 进行排序。

    :param t: 输入 QTensor 。
    :param dim: 排序使用的轴。
    :param descending: 如果是True,进行降序排序,否则使用升序排序。默认为升序。
    :param stable: 是否使用稳定排序,默认为稳定排序。
    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.random.randint(10, size=24).reshape(3,8).astype(np.float32)
        A = QTensor(a)
        AA = tensor.sort(A,1,False)
        print(AA)

        # [
        # [0., 1., 2., 4., 6., 7., 8., 8.],
        # [2., 5., 5., 8., 9., 9., 9., 9.],
        # [1., 2., 5., 5., 5., 6., 7., 7.]
        # ]

argsort
==============================

.. py:function:: pyvqnet.tensor.argsort(t, dim: int, descending=False, stable=True)

    对输入变量沿给定轴进行排序,输出排序好的数据的相应索引。

    :param t: 输入 QTensor 。
    :param dim: 排序使用的轴。
    :param descending: 如果是True,进行降序排序,否则使用升序排序。默认为升序。
    :param stable: 是否使用稳定排序,默认为稳定排序。
    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.random.randint(10, size=24).reshape(3,8).astype(np.float32)
        A = QTensor(a)
        bb = tensor.argsort(A,1,False)
        print(bb)

        # [
        # [4., 0., 1., 7., 5., 3., 2., 6.], 
        #  [3., 0., 7., 6., 2., 1., 4., 5.],
        #  [4., 7., 5., 0., 2., 1., 3., 6.]
        # ]

topK
==============================

.. py:function:: pyvqnet.tensor.topK(t, k, dim=-1, if_descent=True)

    返回给定输入张量沿给定维度的 k 个最大元素。

    如果 if_descent 为 False,则返回 k 个最小元素。

    :param t: 输入 QTensor ,不能是复数或 ``kbool`` 类型。
    :param k: 取排序后的 k 的个数,必须在 [1, dim 维的长度] 之内。
    :param dim: 要排序的维度。默认 = -1,最后一个轴。
    :param if_descent: 排序使用升序还是降序,默认降序。

    :return: 新的 QTensor 。

    ``NaN`` 被视为最大:降序时排在最前,升序时排在最后。
    选出的值可以求梯度,梯度被写回到被选中的位置。

    Example::

        from pyvqnet import kcomplex64
        from pyvqnet.tensor import tensor, QTensor

        x = QTensor([
            24., 13., 15., 4., 3., 8., 11., 3., 6., 15., 24., 13., 15., 3., 3., 8., 7.,
            3., 6., 11.
        ])
        x = x.reshape([2, 5, 1, 2])
        print(tensor.topK(x, 3, 1))

        print(tensor.topK(QTensor([[1.0, float("nan"), 3.0, float("nan"), 2.0]]), 3, -1))

        x.requires_grad = True
        y = tensor.topK(x, 3, 1)
        y.backward(tensor.ones_like(y))
        print(x.grad)

        for call in (
            lambda: tensor.topK(x, 6, 1),
            lambda: tensor.topK(x, 0, 1),
            lambda: tensor.topK(QTensor([[1 + 1j, 2], [3, 4]], dtype=kcomplex64), 1, -1),
        ):
            try:
                call()
            except RuntimeError as e:
                print("RuntimeError:", str(e).split(" ( FILE")[0])

        # [[[[24.,15.]],

        #   [[15.,13.]],

        #   [[11., 8.]]],

        #  [[[24.,13.]],

        #   [[15.,11.]],

        # ...
        # <QTensor [2, 3, 1, 2] DEV_CPU float32>

        # [[nan,nan, 3.]]
        # <QTensor [1, 3] DEV_CPU float32>

        # [[[[1.,1.]],

        #   [[1.,0.]],

        #   [[0.,1.]],

        #   [[1.,0.]],

        #   [[0.,1.]]],

        #  [[[1.,1.]],

        #   [[1.,0.]],

        # ...
        # <QTensor [2, 5, 1, 2] DEV_CPU float32>
        # RuntimeError: selected index k out of range
        # RuntimeError: topk: k=0 would produce a zero-sized dimension, which VQNet tensors do not support
        # RuntimeError: topk does not support complex or bool dtypes

argtopK
==============================

.. py:function:: pyvqnet.tensor.argtopK(t, k, dim=-1, if_descent=True)

    返回给定输入张量沿给定维度的 k 个最大元素的索引。

    如果 if_descent 为 False,则返回 k 个最小元素的索引。

    :param t: 输入 QTensor ,不能是复数或 ``kbool`` 类型。
    :param k: 取排序后的 k 的个数,必须在 [1, dim 维的长度] 之内。
    :param dim: 要排序的维度。默认 = -1,最后一个轴。
    :param if_descent: 排序使用升序还是降序,默认降序。

    :return: 新的 QTensor ,数据类型为 ``kint64``。

    Example::

        from pyvqnet.tensor import tensor, QTensor

        x = QTensor([
            24., 13., 15., 4., 3., 8., 11., 3., 6., 15., 24., 13., 15., 3., 3., 8., 7.,
            3., 6., 11.
        ])
        x = x.reshape([2, 5, 1, 2])
        print(tensor.argtopK(x, 3, 1))
        print(tensor.argtopK(x, 3, 1, False))
        print(tensor.argmax(x, 1))

        # [[[[0,4]],

        #   [[1,0]],

        #   [[3,2]]],

        #  [[[0,0]],

        #   [[1,4]],

        # ...
        # <QTensor [2, 3, 1, 2] DEV_CPU int64>

        # [[[[2,3]],

        #   [[4,1]],

        #   [[3,2]]],

        #  [[[2,3]],

        #   [[4,1]],

        # ...
        # <QTensor [2, 3, 1, 2] DEV_CPU int64>

        # [[[0,4]],

        #  [[0,0]]]
        # <QTensor [2, 1, 2] DEV_CPU int32>

add
==============================

.. py:function:: pyvqnet.tensor.add(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    两个 QTensor 按元素相加。等价于t1 + t2。

    :param t1: 第一个 QTensor 。
    :param t2: 第二个 QTensor 。
    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.add(t1, t2)
        print(x)

        # [5., 7., 9.]

sub
==============================

.. py:function:: pyvqnet.tensor.sub(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    两个 QTensor 按元素相减。等价于t1 - t2。

    :param t1: 第一个 QTensor 。
    :param t2: 第二个 QTensor 。
    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.sub(t1, t2)
        print(x)

        # [-3., -3., -3.]

mul
==============================

.. py:function:: pyvqnet.tensor.mul(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    两个 QTensor 按元素相乘。等价于t1 * t2。

    :param t1: 第一个 QTensor 。
    :param t2: 第二个 QTensor 。
    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.mul(t1, t2)
        print(x)

        # [4., 10., 18.]

divide
==============================

.. py:function:: pyvqnet.tensor.divide(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    两个 QTensor 按元素相除。等价于t1 / t2。

    :param t1: 第一个 QTensor 。
    :param t2: 第二个 QTensor 。
    :return:  输出 QTensor 。


    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.divide(t1, t2)
        print(x)

        # [0.2500000, 0.4000000, 0.5000000]

sums
==============================

.. py:function:: pyvqnet.tensor.sums(t: pyvqnet.tensor.QTensor, dim: Optional[int] = None, keepdim=False)

    对输入的 QTensor 按 dim 设定的轴计算元素和,如果 dim 是None,则返回所有元素和。

    :param t: 输入 QTensor 。
    :param dim: 用于求和的轴,默认为None。
    :param keepdim: 输出张量是否保留了减小的维度。默认为False。
    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor(([1, 2, 3], [4, 5, 6]))
        x = tensor.sums(t)
        print(x)

        # [21.]

cumsum
==============================

.. py:function:: pyvqnet.tensor.cumsum(t, dim=-1)

    返回维度轴中输入元素的累积总和。

    :param t: 输入 QTensor 。不支持 ``kcomplex32`` 类型(调用时抛出 ``RuntimeError``)。
    :param dim: 计算的轴,默认 -1,使用最后一个轴。
    :return: 输出 QTensor 。

    Example::

        from pyvqnet import kcomplex32
        from pyvqnet.tensor import tensor, QTensor
        t = QTensor(([1, 2, 3], [4, 5, 6]))
        x = tensor.cumsum(t,-1)
        print(x)

        c = QTensor([[1 + 2j, 3 - 1j], [0.5 + 0j, -1 + 1j]])
        print(tensor.cumsum(tensor.conj(c), 1))

        try:
            tensor.cumsum(QTensor([[1 + 1j, 2 - 1j]], dtype=kcomplex32), 1)
        except RuntimeError as e:
            print("RuntimeError:", e)

        # [[ 1, 3, 6],
        #  [ 4, 9,15]]
        # <QTensor [2, 3] DEV_CPU int64>

        # [[ 1. -2.j, 4. -1.j],
        #  [ 0.5+0.j,-0.5-1.j]]
        # <QTensor [2, 2] DEV_CPU complex64>
        # RuntimeError: cumsum is not implemented for complex32 storage on CPU.

mean
==============================

.. py:function:: pyvqnet.tensor.mean(t: pyvqnet.tensor.QTensor, dim=None, keepdim=False)

    对输入的 QTensor 按 dim 设定的轴计算元素的平均,如果 dim 是None,则返回所有元素平均。

    :param t: 输入 QTensor ,需要是浮点数或者复数。
    :param dim: 用于求平均的轴,默认为None。
    :param keepdim: 输出张量是否保留了减小的维度。默认为False。
    :return: 输出 QTensor 或 均值。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([[1, 2, 3], [4, 5, 6.0]])
        x = tensor.mean(t, dim=1)
        print(x)

        # [2. 5.]

median
==============================

.. py:function:: pyvqnet.tensor.median(t: pyvqnet.tensor.QTensor, dim=None, keepdim=False)

    对输入的 QTensor 按 dim 设定的轴计算元素的中值,如果 dim 是None,则返回所有元素的中值。

    :param t: 输入 QTensor ,不支持复数类型。
    :param dim: 用于求中值的轴,默认为None。
    :param keepdim: 输出张量是否保留了减小的维度。默认为False。
    :return: 输出 QTensor 或 中值。

    当 ``dim`` 为 ``None`` 且张量不含元素时,抛出 ``ValueError``。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1.5219, -1.5212,  0.2202]])
        median_a = tensor.median(a)
        print(median_a)

        b = QTensor([[0.2505, -0.3982, -0.9948,  0.3518, -1.3131],
                     [0.3180, -0.6993,  1.0436,  0.0438,  0.2270],
                     [-0.2751,  0.7303,  0.2192,  0.3321,  0.2488],
                     [1.0778, -1.9510,  0.7048,  0.4742, -0.7125]])
        median_b = tensor.median(b,1, False)
        print(median_b)

        try:
            tensor.median(QTensor([]))
        except ValueError as e:
            print("ValueError:", e)

        # 0.2202
        # <QTensor [] DEV_CPU float32>

        # [-0.3982, 0.227 , 0.2488, 0.4742]
        # <QTensor [4] DEV_CPU float32>
        # ValueError: median(): cannot reduce a tensor with zero elements

std
==============================

.. py:function:: pyvqnet.tensor.std(t: pyvqnet.tensor.QTensor, dim=None, keepdim=False, unbiased=True)

    对输入的 QTensor 按 dim 设定的轴计算元素的标准差,如果 dim 是None,则返回所有元素标准差。

    输入为复数类型时,结果是与实部精度对应的实数类型:``kcomplex64`` 的结果为 ``kfloat32``, ``kcomplex32`` 的结果为 ``kfloat16``。

    :param t: 输入 QTensor 。
    :param dim: 用于求标准差的轴,默认为None。
    :param keepdim: 输出张量是否保留了减小的维度。默认为False。
    :param unbiased: 是否使用贝塞尔修正,默认使用。
    :return: 输出 QTensor 或 标准差。

    Example::

        from pyvqnet import kcomplex32
        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[-0.8166, -1.3802, -0.3560]])
        std_a = tensor.std(a)
        print(std_a)

        b = QTensor([[0.2505, -0.3982, -0.9948,  0.3518, -1.3131],
                     [0.3180, -0.6993,  1.0436,  0.0438,  0.2270],
                     [-0.2751,  0.7303,  0.2192,  0.3321,  0.2488],
                     [1.0778, -1.9510,  0.7048,  0.4742, -0.7125]])
        std_b = tensor.std(b, 1, False, False)
        print(std_b)

        c = QTensor([1 + 2j, 0.5 - 1j, -0.25 + 0.75j], dtype=kcomplex32)
        print(tensor.std(c).dtype, tensor.std(c))
        print(tensor.std(c.astype(__import__("pyvqnet").kcomplex64)).dtype)

        # 0.5129625
        # <QTensor [] DEV_CPU float32>

        # [0.6593543,0.5583113,0.3206565,1.1103367]
        # <QTensor [4] DEV_CPU float32>
        # float16 
        # 1.633
        # <QTensor [] DEV_CPU float16>
        # float32

var
==============================

.. py:function:: pyvqnet.tensor.var(t: pyvqnet.tensor.QTensor, dim=None, keepdim=False, unbiased=True)

    对输入的 QTensor 按 dim 设定的轴计算元素的方差,如果 dim 是None,则返回所有元素方差。

    输入为复数类型时,结果是与实部精度对应的实数类型:``kcomplex64`` 的结果为 ``kfloat32``, ``kcomplex32`` 的结果为 ``kfloat16``。

    :param t: 输入 QTensor 。
    :param dim: 用于求方差的轴,默认为None。
    :param keepdim: 输出张量是否保留了减小的维度。默认为False。
    :param unbiased: 是否使用贝塞尔修正,默认使用。
    :return: 输出 QTensor 或方差。

    Example::

        from pyvqnet import kcomplex32, kcomplex64
        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[-0.8166, -1.3802, -0.3560]])
        a_var = tensor.var(a)
        print(a_var)

        b = QTensor([[0.2505, -0.3982, -0.9948,  0.3518, -1.3131],
                     [0.3180, -0.6993,  1.0436,  0.0438,  0.2270],
                     [-0.2751,  0.7303,  0.2192,  0.3321,  0.2488],
                     [1.0778, -1.9510,  0.7048,  0.4742, -0.7125]])
        print(tensor.var(b, 1, False, False))

        c = QTensor([1 + 2j, 0.5 - 1j, -0.25 + 0.75j], dtype=kcomplex32)
        print(tensor.var(c).dtype, tensor.var(c))
        print(tensor.var(c.astype(kcomplex64)).dtype)

        # 0.2631305
        # <QTensor [] DEV_CPU float32>

        # [0.4347481,0.3117115,0.1028206,1.2328476]
        # <QTensor [4] DEV_CPU float32>
        # float16 
        # 2.666
        # <QTensor [] DEV_CPU float16>
        # float32

matmul
==============================

.. py:function:: pyvqnet.tensor.matmul(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    二维矩阵点乘或3、4维张量进行批矩阵乘法,或一维向量与二维矩阵矩阵向量积,或两个一维向量点积。

    两个操作数的数据类型不同时,先把它们转换成较宽的类型再相乘:含复数操作数时转换成对应的复数类型
    (``kcomplex128`` > ``kcomplex64`` > ``kcomplex32``),两个实数操作数转换成 ``kfloat64`` 或 ``kfloat32``。
    两个操作数是不同的整数或 ``kbool`` 类型时抛出 ``TypeError``。

    :param t1: 第一个 QTensor 。
    :param t2: 第二个 QTensor 。
    :return:  输出 QTensor 。

    Example::

        from pyvqnet import kfloat16, kint32, kint64
        from pyvqnet.tensor import tensor

        t1 = tensor.ones([2, 3])
        t1.requires_grad = True
        t2 = tensor.ones([3, 4])
        t2.requires_grad = True
        t3 = tensor.matmul(t1, t2)
        t3.backward(tensor.ones_like(t3))
        print(t1.grad)
        print(t2.grad)

        a = tensor.ones([2, 3], dtype=kfloat16)
        b = tensor.ones([3, 4])
        print(tensor.matmul(a, b).dtype)
        try:
            tensor.matmul(tensor.ones([2, 3], dtype=kint32), tensor.ones([3, 4], dtype=kint64))
        except TypeError as e:
            print("TypeError:", e)

        # [[4.,4.,4.],
        #  [4.,4.,4.]]
        # <QTensor [2, 3] DEV_CPU float32>

        # [[2.,2.,2.,2.],
        #  [2.,2.,2.,2.],
        #  [2.,2.,2.,2.]]
        # <QTensor [3, 4] DEV_CPU float32>
        # float32
        # TypeError: matmul() does not support mixed integer/bool dtypes: int32 and int64

kron
==============================

.. py:function:: pyvqnet.tensor.kron(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    计算 ``t1`` 和  ``t2`` 的 Kronecker 积,用 :math:`\otimes` 表示。

    如果 ``t1`` 是一个 :math:`(a_0 \times a_1 \times \dots \times a_n)` 张量并且 ``t2`` 是一个 :math:`(b_0 \times b_1 \times \dots \times b_n)` 张量,结果将是 :math:`(a_0*b_0 \times a_1*b_1 \times \dots \times a_n*b_n)` 张量,包含以下条目:

     .. math::
         (\text{input} \otimes \text{other})_{k_0, k_1, \dots, k_n} =
             \text{input}_{i_0, i_1, \dots, i_n} * \text{other}_{j_0, j_1, \dots, j_n},

     其中 :math:`k_t = i_t * b_t + j_t` 为 :math:`0 \leq t \leq n`。
     如果一个张量的维数少于另一个,它将被解压缩,直到它具有相同的维数。

    :param t1: 第一个 QTensor 。
    :param t2: 第二个 QTensor 。
    :return:  输出 QTensor 。

    Example::

        from pyvqnet import tensor
        a = tensor.arange(1,1+ 24).reshape([2,1,2,3,2])
        b = tensor.arange(1,1+ 24).reshape([6,4])


        c = tensor.kron(a,b)
        print(c)

        # [[[[[  1.   2.   3.   4.   2.   4.   6.   8.]
        #     [  5.   6.   7.   8.  10.  12.  14.  16.]
        #     [  9.  10.  11.  12.  18.  20.  22.  24.]
        #     [ 13.  14.  15.  16.  26.  28.  30.  32.]
        #     [ 17.  18.  19.  20.  34.  36.  38.  40.]
        #     [ 21.  22.  23.  24.  42.  44.  46.  48.]
        #     [  3.   6.   9.  12.   4.   8.  12.  16.]
        #     [ 15.  18.  21.  24.  20.  24.  28.  32.]
        #     [ 27.  30.  33.  36.  36.  40.  44.  48.]
        #     [ 39.  42.  45.  48.  52.  56.  60.  64.]
        #     [ 51.  54.  57.  60.  68.  72.  76.  80.]
        #     [ 63.  66.  69.  72.  84.  88.  92.  96.]
        #     [  5.  10.  15.  20.   6.  12.  18.  24.]
        #     [ 25.  30.  35.  40.  30.  36.  42.  48.]
        #     [ 45.  50.  55.  60.  54.  60.  66.  72.]
        #     [ 65.  70.  75.  80.  78.  84.  90.  96.]
        #     [ 85.  90.  95. 100. 102. 108. 114. 120.]
        #     [105. 110. 115. 120. 126. 132. 138. 144.]]

        #    [[  7.  14.  21.  28.   8.  16.  24.  32.]
        #     [ 35.  42.  49.  56.  40.  48.  56.  64.]
        #     [ 63.  70.  77.  84.  72.  80.  88.  96.]
        #     [ 91.  98. 105. 112. 104. 112. 120. 128.]
        #     [119. 126. 133. 140. 136. 144. 152. 160.]
        #     [147. 154. 161. 168. 168. 176. 184. 192.]
        #     [  9.  18.  27.  36.  10.  20.  30.  40.]
        #     [ 45.  54.  63.  72.  50.  60.  70.  80.]
        #     [ 81.  90.  99. 108.  90. 100. 110. 120.]
        #     [117. 126. 135. 144. 130. 140. 150. 160.]
        #     [153. 162. 171. 180. 170. 180. 190. 200.]
        #     [189. 198. 207. 216. 210. 220. 230. 240.]
        #     [ 11.  22.  33.  44.  12.  24.  36.  48.]
        #     [ 55.  66.  77.  88.  60.  72.  84.  96.]
        #     [ 99. 110. 121. 132. 108. 120. 132. 144.]
        #     [143. 154. 165. 176. 156. 168. 180. 192.]
        #     [187. 198. 209. 220. 204. 216. 228. 240.]
        #     [231. 242. 253. 264. 252. 264. 276. 288.]]]]



        #  [[[[ 13.  26.  39.  52.  14.  28.  42.  56.]
        #     [ 65.  78.  91. 104.  70.  84.  98. 112.]
        #     [117. 130. 143. 156. 126. 140. 154. 168.]
        #     [169. 182. 195. 208. 182. 196. 210. 224.]
        #     [221. 234. 247. 260. 238. 252. 266. 280.]
        #     [273. 286. 299. 312. 294. 308. 322. 336.]
        #     [ 15.  30.  45.  60.  16.  32.  48.  64.]
        #     [ 75.  90. 105. 120.  80.  96. 112. 128.]
        #     [135. 150. 165. 180. 144. 160. 176. 192.]
        #     [195. 210. 225. 240. 208. 224. 240. 256.]
        #     [255. 270. 285. 300. 272. 288. 304. 320.]
        #     [315. 330. 345. 360. 336. 352. 368. 384.]
        #     [ 17.  34.  51.  68.  18.  36.  54.  72.]
        #     [ 85. 102. 119. 136.  90. 108. 126. 144.]
        #     [153. 170. 187. 204. 162. 180. 198. 216.]
        #     [221. 238. 255. 272. 234. 252. 270. 288.]
        #     [289. 306. 323. 340. 306. 324. 342. 360.]
        #     [357. 374. 391. 408. 378. 396. 414. 432.]]

        #    [[ 19.  38.  57.  76.  20.  40.  60.  80.]
        #     [ 95. 114. 133. 152. 100. 120. 140. 160.]
        #     [171. 190. 209. 228. 180. 200. 220. 240.]
        #     [247. 266. 285. 304. 260. 280. 300. 320.]
        #     [323. 342. 361. 380. 340. 360. 380. 400.]
        #     [399. 418. 437. 456. 420. 440. 460. 480.]
        #     [ 21.  42.  63.  84.  22.  44.  66.  88.]
        #     [105. 126. 147. 168. 110. 132. 154. 176.]
        #     [189. 210. 231. 252. 198. 220. 242. 264.]
        #     [273. 294. 315. 336. 286. 308. 330. 352.]
        #     [357. 378. 399. 420. 374. 396. 418. 440.]
        #     [441. 462. 483. 504. 462. 484. 506. 528.]
        #     [ 23.  46.  69.  92.  24.  48.  72.  96.]
        #     [115. 138. 161. 184. 120. 144. 168. 192.]
        #     [207. 230. 253. 276. 216. 240. 264. 288.]
        #     [299. 322. 345. 368. 312. 336. 360. 384.]
        #     [391. 414. 437. 460. 408. 432. 456. 480.]
        #     [483. 506. 529. 552. 504. 528. 552. 576.]]]]]


einsum
==============================

.. py:function:: pyvqnet.tensor.einsum(equation, *operands)

    使用基于爱因斯坦求和约定的符号沿指定的维度对输入操作数元素的乘积求和。

    .. note::

        此函数使用 opt_einsum (https://optimized-einsum.readthedocs.io/en/stable/) 来加速计算或通过优化收缩顺序来减少内存消耗。当至少有三个输入时,会发生此优化。

        对于更加复杂的 `einsum` ,可另外导入opt_einsum直接对QTensor进行计算。

    :param equation: 爱因斯坦求和的下标。

    :param operands: 要计算爱因斯坦求和的张量。

    :return:
            QTensor 结果。

    Example::

        from pyvqnet import tensor

        vqneta = tensor.randn((3, 5, 4))
        vqnetl = tensor.randn((2, 5))
        vqnetr = tensor.randn((2, 4))
        z = tensor.einsum('bn,anm,bm->ba',  vqnetl, vqneta,vqnetr)
        print(z.shape)
        #[2, 3]
        vqneta = tensor.randn((20,30,40,50))
        z = tensor.einsum('...ij->...ji', vqneta)
        print(z.shape)
        #[20, 30, 50, 40]

reciprocal
==============================

.. py:function:: pyvqnet.tensor.reciprocal(t)

    计算输入 QTensor 的倒数。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(1, 10, 1)
        u = tensor.reciprocal(t)
        print(u)

        #[1., 0.5000000, 0.3333333, 0.2500000, 0.2000000, 0.1666667, 0.1428571, 0.1250000, 0.1111111]

sign
==============================

.. py:function:: pyvqnet.tensor.sign(t)

    对输入 t 中每个元素进行正负判断,并且输出正负判断值:1代表正,-1代表负,0代表零。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。


    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = tensor.arange(-5, 5, 1)
        u = tensor.sign(t)
        print(u)

        # [-1., -1., -1., -1., -1., 0., 1., 1., 1., 1.]

neg
==============================

.. py:function:: pyvqnet.tensor.neg(t: pyvqnet.tensor.QTensor)

    计算输入 t 每个元素的相反数并返回。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.neg(t)
        print(x)

        # [-1., -2., -3.]

trace
==============================

.. py:function:: pyvqnet.tensor.trace(t, k: int = 0)

    返回二维矩阵的迹。

    :param t: 输入 QTensor 。
    :param k: 偏移量(主对角线为 0,正数为向上偏移,负数为向下偏移),默认为0。

    :return: 输入二维矩阵的对角线元素之和。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = tensor.randn([4,4])
        for k in range(-3, 4):
            u=tensor.trace(t,k=k)
            print(u)

        # 0.07717618346214294
        # -1.9287869930267334
        # 0.6111435890197754
        # 2.8094992637634277
        # 0.6388946771621704
        # -1.3400784730911255
        # 0.26980453729629517

exp
==============================

.. py:function:: pyvqnet.tensor.exp(t: pyvqnet.tensor.QTensor)

    计算输入 t 每个元素的自然数e为底指数。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.exp(t)
        print(x)

        # [2.7182817, 7.3890562, 20.0855369]

acos
==============================

.. py:function:: pyvqnet.tensor.acos(t: pyvqnet.tensor.QTensor)

    计算输入 t 每个元素的反余弦。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.arange(36).reshape(2,6,3).astype(np.float32)
        a =a/100
        A = QTensor(a,requires_grad = True)
        y = tensor.acos(A)
        print(y)

        # [
        # [[1.5707964, 1.5607961, 1.5507950],
        #  [1.5407919, 1.5307857, 1.5207754],
        #  [1.5107603, 1.5007390, 1.4907107],
        #  [1.4806744, 1.4706289, 1.4605733],
        #  [1.4505064, 1.4404273, 1.4303349],
        #  [1.4202280, 1.4101057, 1.3999666]],
        # [[1.3898098, 1.3796341, 1.3694384],
        #  [1.3592213, 1.3489819, 1.3387187],
        #  [1.3284305, 1.3181161, 1.3077742],
        #  [1.2974033, 1.2870022, 1.2765695],
        #  [1.2661036, 1.2556033, 1.2450669],
        #  [1.2344928, 1.2238795, 1.2132252]]
        # ]

asin
==============================

.. py:function:: pyvqnet.tensor.asin(t: pyvqnet.tensor.QTensor)

    计算输入 t 每个元素的反正弦。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = tensor.arange(-1, 1, .5)
        u = tensor.asin(t)
        print(u)

        #[-1.5707964, -0.5235988, 0., 0.5235988]

atan
==============================

.. py:function:: pyvqnet.tensor.atan(t: pyvqnet.tensor.QTensor)

    计算输入 t 每个元素的反正切。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-1, 1, .5)
        u = tensor.atan(t)
        print(u)

        # [-0.7853981, -0.4636476, 0., 0.4636476]

sin
==============================

.. py:function:: pyvqnet.tensor.sin(t: pyvqnet.tensor.QTensor)

    计算输入 t 每个元素的正弦。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.sin(t)
        print(x)

        # [0.8414709, 0.9092974, 0.1411200]

cos
==============================

.. py:function:: pyvqnet.tensor.cos(t: pyvqnet.tensor.QTensor)

    计算输入 t 每个元素的余弦。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.cos(t)
        print(x)

        # [0.5403022, -0.4161468, -0.9899924]

tan 
==============================

.. py:function:: pyvqnet.tensor.tan(t: pyvqnet.tensor.QTensor)

    计算输入 t 每个元素的正切。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.tan(t)
        print(x)

        # [1.5574077, -2.1850397, -0.1425465]

tanh
==============================

.. py:function:: pyvqnet.tensor.tanh(t: pyvqnet.tensor.QTensor)

    计算输入 t 每个元素的双曲正切。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.tanh(t)
        print(x)

        # [0.7615941, 0.9640275, 0.9950547]

sinh
==============================

.. py:function:: pyvqnet.tensor.sinh(t: pyvqnet.tensor.QTensor)

    计算输入 t 每个元素的双曲正弦。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.sinh(t)
        print(x)

        # [1.1752011, 3.6268603, 10.0178747]

cosh
==============================

.. py:function:: pyvqnet.tensor.cosh(t: pyvqnet.tensor.QTensor)

    计算输入 t 每个元素的双曲余弦。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.cosh(t)
        print(x)

        # [1.5430806, 3.7621955, 10.0676622]

power
==============================

.. py:function:: pyvqnet.tensor.power(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    第一个 QTensor 的元素计算第二个 QTensor 的幂指数。

    :param t1: 第一个 QTensor 。
    :param t2: 第二个 QTensor 。
    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([1, 4, 3])
        t2 = QTensor([2, 5, 6])
        x = tensor.power(t1, t2)
        print(x)

        # [1., 1024., 729.]

abs
==============================

.. py:function:: pyvqnet.tensor.abs(t: pyvqnet.tensor.QTensor)

    计算输入 QTensor 的每个元素的绝对值。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, -2, 3])
        x = tensor.abs(t)
        print(x)

        # [1., 2., 3.]

log
==============================

.. py:function:: pyvqnet.tensor.log(t: pyvqnet.tensor.QTensor)

    计算输入 QTensor 的每个元素的自然对数值。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.log(t)
        print(x)

        # [0., 0.6931471, 1.0986123]

log_softmax
==============================

.. py:function:: pyvqnet.tensor.log_softmax(t, dim=-1)

    顺序计算在dim维度上的softmax函数以及log函数的结果。

    :param t: 输入 QTensor 。
    :param dim: 用于求softmax的轴,默认为-1。

    :return: 输出 QTensor。

    Example::

        from pyvqnet import tensor
        output = tensor.arange(1,13).reshape([3,2,2])
        t = tensor.log_softmax(output,1)
        print(t)
        # [
        # [[-2.1269281, -2.1269281],
        #  [-0.1269280, -0.1269280]],
        # [[-2.1269281, -2.1269281],
        #  [-0.1269280, -0.1269280]],
        # [[-2.1269281, -2.1269281],
        #  [-0.1269280, -0.1269280]]
        # ]

sqrt
==============================

.. py:function:: pyvqnet.tensor.sqrt(t: pyvqnet.tensor.QTensor)

    计算输入 QTensor 的每个元素的平方根值。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.sqrt(t)
        print(x)

        # [1., 1.4142135, 1.7320507]

square
==============================

.. py:function:: pyvqnet.tensor.square(t: pyvqnet.tensor.QTensor)

    计算输入 QTensor 的每个元素的平方值。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.square(t)
        print(x)

        # [1., 4., 9.]


eigh
==============================

.. py:function:: pyvqnet.tensor.eigh(t: QTensor)
 
    返回复厄米矩阵(共轭对称)或实对称矩阵的特征值和特征向量。
    返回两个对象,一个包含a的特征值的一维数组,
    以及相应特征向量(以列表示)的二维方阵或矩阵(取决于输入类型)。

    :param: 输入QTensor。
    :param: t的特征值和特征向量。
    :return:

        返回特征值以及特征向量

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet import tensor


        def generate_random_symmetric_matrix(n):
                A = pyvqnet.tensor.randn((n, n))
                A = A + A.transpose()
                return A

        n = 3
        symmetric_matrix = generate_random_symmetric_matrix(n)

        evs,vecs = pyvqnet.tensor.eigh(symmetric_matrix)
        print(evs)
        print(vecs)
        # [-4.0669565,-1.9191254,-1.3642329]
        # <QTensor [3] DEV_CPU kfloat32>

        # [[-0.9889652, 0.0325959,-0.1445187],
        #  [ 0.0912495, 0.9025176,-0.4208745],
        #  [ 0.1167119,-0.4294176,-0.8955328]]
        # <QTensor [3, 3] DEV_CPU kfloat32>


frobenius_norm
==============================

.. py:function:: pyvqnet.tensor.frobenius_norm(t: QTensor, dim: int = None, keepdim=False)

    对输入的 QTensor 按 dim 设定的轴计算张量的F范数,如果 dim 是None,则返回所有元素F范数。

    :param t: 输入 QTensor 。
    :param dim: 用于求F范数的轴,默认为None。
    :param keepdim: 输出张量是否保留了减小的维度。默认为False。
    :return: 输出 QTensor 或 F范数值。


    Example::

        from pyvqnet.tensor import tensor,QTensor
        t = QTensor([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]],
                    [[13., 14., 15.], [16., 17., 18.]]])
        t.requires_grad = True
        result = tensor.frobenius_norm(t, -2, False)
        print(result)
        # [
        # [4.1231055, 5.3851647, 6.7082038],
        #  [12.2065554, 13.6014709, 15.],
        #  [20.6155281, 22.0227146, 23.4307499]
        # ]


逻辑函数
*************************************

maximum
==============================

.. py:function:: pyvqnet.tensor.maximum(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    计算两个 QTensor 的逐元素中的较大值。

    :param t1: 第一个 QTensor 。
    :param t2: 第二个 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([6, 4, 3])
        t2 = QTensor([2, 5, 7])
        x = tensor.maximum(t1, t2)
        print(x)

        # [6., 5., 7.]

minimum
==============================

.. py:function:: pyvqnet.tensor.minimum(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    计算两个 QTensor 的逐元素中的较小值。

    :param t1: 第一个 QTensor 。
    :param t2: 第二个 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([6, 4, 3])
        t2 = QTensor([2, 5, 7])
        x = tensor.minimum(t1, t2)
        print(x)

        # [2., 4., 3.]

min
==============================

.. py:function:: pyvqnet.tensor.min(t: pyvqnet.tensor.QTensor, dim=None, keepdim=False)

    对输入的 QTensor 按 dim 设定的轴计算元素的最小值,如果 dim 是None,则返回所有元素的最小值。

    :param t: 输入 QTensor ,不支持复数类型。
    :param dim: 用于求最小值的轴,默认为None。
    :param keepdim: 输出张量是否保留了减小的维度。默认为False。

    :return: 输出 QTensor 或浮点数。

    当 ``dim`` 为 ``None`` 且张量不含元素时,抛出 ``ValueError``。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([[1., 2., 3.], [4., 5., 6.]])
        x = tensor.min(t, dim=1, keepdim=True)
        print(x)
        print(tensor.min(t))

        t.requires_grad = True
        y = tensor.min(t, dim=1, keepdim=False)
        y.backward(tensor.ones_like(y))
        print(t.grad)

        try:
            tensor.min(QTensor([]))
        except ValueError as e:
            print("ValueError:", e)

        # [[1.],
        #  [4.]]
        # <QTensor [2, 1] DEV_CPU float32>

        # 1.
        # <QTensor [] DEV_CPU float32>

        # [[1.,0.,0.],
        #  [1.,0.,0.]]
        # <QTensor [2, 3] DEV_CPU float32>
        # ValueError: min(): cannot reduce a tensor with zero elements

max
==============================

.. py:function:: pyvqnet.tensor.max(t: pyvqnet.tensor.QTensor, dim=None, keepdim=False)

    对输入的 QTensor 按 dim 设定的轴计算元素的最大值,如果 dim 是None,则返回所有元素的最大值。

    :param t: 输入 QTensor ,不支持复数类型。
    :param dim: 用于求最大值的轴,默认为None。
    :param keepdim: 输出张量是否保留了减小的维度。默认为False。

    :return: 输出 QTensor 或浮点数。

    当 ``dim`` 为 ``None`` 且张量不含元素时,抛出 ``ValueError``。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([[1., 2., 3.], [4., 5., 6.]])
        x = tensor.max(t, dim=1, keepdim=True)
        print(x)
        print(tensor.max(t))

        t.requires_grad = True
        y = tensor.max(t, dim=1, keepdim=False)
        y.backward(tensor.ones_like(y))
        print(t.grad)

        try:
            tensor.max(QTensor([]))
        except ValueError as e:
            print("ValueError:", e)

        # [[3.],
        #  [6.]]
        # <QTensor [2, 1] DEV_CPU float32>

        # 6.
        # <QTensor [] DEV_CPU float32>

        # [[0.,0.,1.],
        #  [0.,0.,1.]]
        # <QTensor [2, 3] DEV_CPU float32>
        # ValueError: max(): cannot reduce a tensor with zero elements

clip
==============================

.. py:function:: pyvqnet.tensor.clip(t: pyvqnet.tensor.QTensor, min_val, max_val)

    将输入的所有元素进行剪裁,使得输出元素限制在[min_val, max_val]。

    :param t: 输入 QTensor 。
    :param min_val:  裁剪下限值。
    :param max_val:  裁剪上限值。
    :return:  output QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([2, 4, 6])
        x = tensor.clip(t, 3, 8)
        print(x)

        # [3., 4., 6.]


where
==============================

.. py:function:: pyvqnet.tensor.where(condition: pyvqnet.tensor.QTensor, t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)


    根据条件返回从 t1 或 t2 中选择的元素。

    :param condition: 判断条件 QTensor,需要是kbool数据类型 。
    :param t1: 如果满足条件,则从中获取元素。
    :param t2: 如果条件不满足,则从中获取元素。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.where(t1 < 2, t1, t2)
        print(x)

        # [1., 5., 6.]

nonzero
==============================

.. py:function:: pyvqnet.tensor.nonzero(t)

    返回一个包含非零元素索引的 QTensor 。

    :param t: 输入 QTensor 。
    :return: 输出 QTensor 包含非零元素的索引。

    Example::
    
        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([[0.6, 0.0, 0.0, 0.0],
                                    [0.0, 0.4, 0.0, 0.0],
                                    [0.0, 0.0, 1.2, 0.0],
                                    [0.0, 0.0, 0.0,-0.4]])
        t = tensor.nonzero(t)
        print(t)
        # [
        # [0., 0.],
        # [1., 1.],
        # [2., 2.],
        # [3., 3.]
        # ]

isfinite
==============================

.. py:function:: pyvqnet.tensor.isfinite(t)

    逐元素判断输入是否为Finite (既非 +/-INF 也非 +/-NaN )。

    :param t: 输入 QTensor 。
    :return: 输出 QTensor , 其中对应位置元素满足条件时返回True,否则返回False。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isfinite(t)
        print(flag)

        #[ True False  True False False]

isinf
==============================

.. py:function:: pyvqnet.tensor.isinf(t)

    逐元素判断输入的每一个值是否为 +/-INF 。

    :param t: 输入 QTensor 。
    :return: 输出 QTensor , 其中对应位置元素满足条件时返回True,否则返回False。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isinf(t)
        print(flag)

        # [False  True False  True False]

isnan
==============================

.. py:function:: pyvqnet.tensor.isnan(t)

    逐元素判断输入的每一个值是否为 +/-NaN 。

    :param t: 输入 QTensor 。
    :return: 输出 QTensor , 其中对应位置元素满足条件时返回True,否则返回False。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isnan(t)
        print(flag)

        # [False False False False  True]

isneginf
==============================

.. py:function:: pyvqnet.tensor.isneginf(t)

    逐元素判断输入的每一个值是否为 -INF 。

    :param t: 输入 QTensor 。
    :return: 输出 QTensor , 其中对应位置元素满足条件时返回True,否则返回False。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isneginf(t)
        print(flag)

        # [False False False  True False]

isposinf
==============================

.. py:function:: pyvqnet.tensor.isposinf(t)

    逐元素判断输入的每一个值是否为 +INF 。

    :param t: 输入 QTensor 。
    :return: 输出 QTensor , 其中对应位置元素满足条件时返回True,否则返回False。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isposinf(t)
        print(flag)

        # [False  True False False False]

logical_and
==============================

.. py:function:: pyvqnet.tensor.logical_and(t1, t2)

    对两个输入进行逐元素逻辑与操作,其中对应位置元素满足条件时返回True,否则返回False。

    :param t1: 输入 QTensor 。
    :param t2: 输入 QTensor 。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([0, 1, 10, 0])
        b = QTensor([4, 0, 1, 0])
        flag = tensor.logical_and(a,b)
        print(flag)

        # [False False  True False]

logical_or
==============================

.. py:function:: pyvqnet.tensor.logical_or(t1, t2)

    对两个输入进行逐元素逻辑或操作,其中对应位置元素满足条件时返回True,否则返回False。

    :param t1: 输入 QTensor 。
    :param t2: 输入 QTensor 。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([0, 1, 10, 0])
        b = QTensor([4, 0, 1, 0])
        flag = tensor.logical_or(a,b)
        print(flag)

        # [ True  True  True False]

logical_not
==============================

.. py:function:: pyvqnet.tensor.logical_not(t)

    对输入进行逐元素逻辑非操作,其中对应位置元素满足条件时返回True,否则返回False。

    :param t: 输入 QTensor 。
    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([0, 1, 10, 0])
        flag = tensor.logical_not(a)
        print(flag)

        # [ True False False  True]

logical_xor
==============================

.. py:function:: pyvqnet.tensor.logical_xor(t1, t2)

    对两个输入进行逐元素逻辑异或操作,其中对应位置元素满足条件时返回True,否则返回False。

    :param t1: 输入 QTensor 。
    :param t2: 输入 QTensor 。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([0, 1, 10, 0])
        b = QTensor([4, 0, 1, 0])
        flag = tensor.logical_xor(a,b)
        print(flag)

        # [ True  True False False]

greater
==============================

.. py:function:: pyvqnet.tensor.greater(t1, t2)

    逐元素比较 t1 是否大于 t2 ,其中对应位置元素满足条件时返回True,否则返回False。

    :param t1: 输入 QTensor 。
    :param t2: 输入 QTensor 。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.greater(a,b)
        print(flag)

        # [[False  True]
        #  [False False]]

greater_equal
==============================

.. py:function:: pyvqnet.tensor.greater_equal(t1, t2)

    逐元素比较 t1 是否大于等于 t2 ,其中对应位置元素满足条件时返回True,否则返回False。

    :param t1: 输入 QTensor 。
    :param t2: 输入 QTensor 。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.greater_equal(a,b)
        print(flag)

        #[[ True  True]
        # [False  True]]

less
==============================

.. py:function:: pyvqnet.tensor.less(t1, t2)

    逐元素比较 t1 是否小于 t2 ,其中对应位置元素满足条件时返回True,否则返回False。

    :param t1: 输入 QTensor 。
    :param t2: 输入 QTensor 。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.less(a,b)
        print(flag)

        #[[False False]
        # [ True False]]

less_equal
==============================

.. py:function:: pyvqnet.tensor.less_equal(t1, t2)

    逐元素比较 t1 是否小于等于 t2 ,其中对应位置元素满足条件时返回True,否则返回False。

    :param t1: 输入 QTensor 。
    :param t2: 输入 QTensor 。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.less_equal(a,b)
        print(flag)


        # [[ True False]
        #  [ True  True]]

equal
==============================

.. py:function:: pyvqnet.tensor.equal(t1, t2)

    逐元素比较 t1 是否等于 t2 ,其中对应位置元素满足条件时返回True,否则返回False。

    :param t1: 输入 QTensor 。
    :param t2: 输入 QTensor 。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.equal(a,b)
        print(flag)

        #[[ True False]
        # [False  True]]

not_equal
==============================

.. py:function:: pyvqnet.tensor.not_equal(t1, t2)

    逐元素比较 t1 是否不等于 t2 ,其中对应位置元素满足条件时返回True,否则返回False。

    :param t1: 输入 QTensor 。
    :param t2: 输入 QTensor 。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.not_equal(a,b)
        print(flag)

        #[[False  True]
        # [ True False]]



bitwise_and
==============================

.. py:function:: pyvqnet.tensor.bitwise_and(t1, t2)
 
    逐元素计算两个 QTensor 的按位与。

    :param t1: 输入 QTensor t1。只有整数或布尔值才是有效输入。
    :param t2: 输入 QTensor t2。只有整数或布尔值才是有效输入。

    Example::

        from pyvqnet.tensor import *
        import numpy as np
        from pyvqnet.dtype import *
        powers_of_two = 1 << np.arange(14, dtype=np.int64)[::-1]
        samples = tensor.QTensor([23],dtype=kint8)
        samples = samples.unsqueeze(-1)
        states_sampled_base_ten = samples & tensor.QTensor(powers_of_two,dtype = samples.dtype, device = samples.device)
        print(states_sampled_base_ten)
        #[[ 0, 0, 0, 0, 0, 0, 0, 0, 0,16, 0, 4, 2, 1]]


变换函数
*********************

broadcast
==============================

.. py:function:: pyvqnet.tensor.broadcast(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    受到某些限制,较小的阵列在整个更大的阵列,以便它们具有兼容的形状。该接口可对入参张量进行自动微分。

    参考https://numpy.org/doc/stable/user/basics.broadcasting.html

    :param t1: 输入 QTensor 1
    :param t2: 输入 QTensor 2

    :return t11: 具有新的广播形状 t1。
    :return t22: 具有新广播形状的 t2。

    Example::

        from pyvqnet.tensor import *
        t1 = ones([5,4])
        t2 = ones([4])

        t11, t22 = tensor.broadcast(t1, t2)

        print(t11.shape)
        print(t22.shape)


        t1 = ones([5,4])
        t2 = ones([1])

        t11, t22 = tensor.broadcast(t1, t2)

        print(t11.shape)
        print(t22.shape)


        t1 = ones([5,4])
        t2 = ones([2,1,4])

        t11, t22 = tensor.broadcast(t1, t2)

        print(t11.shape)
        print(t22.shape)


        # [5, 4]
        # [5, 4]
        # [5, 4]
        # [5, 4]
        # [2, 5, 4]
        # [2, 5, 4]

select
==============================

.. py:function:: pyvqnet.tensor.select(t: pyvqnet.tensor.QTensor, index)

    输入字符串形式的索引位置,获取该索引下的数据切片,返回一个新的 QTensor 。
    
    :param t: 输入 QTensor 。
    :param index: 一个字符串包含切片的索引。
    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        t = QTensor(np.arange(1,25.0).reshape(2,3,4))
              
        indx = [":", "0", ":"]        
        t.requires_grad = True
        t.zero_grad()
        ts = tensor.select(t,indx)
        
        print(ts)  
        # [
        # [[1., 2., 3., 4.]],
        # [[13., 14., 15., 16.]]
        # ]

concatenate
==============================

.. py:function:: pyvqnet.tensor.concatenate(args: list, dim=0)

    对 args 内的多个 QTensor 沿 dim 维度进行联结,返回一个新的 QTensor 。

    :param args: 包含输入 QTensor 。
    :param dim: 要连接的维度。 必须介于 0 和输入张量的最大维数之间。
    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        x = QTensor([[1.0, 2, 3],[4,5,6]], requires_grad=True)
        y = 1-x
        x = tensor.concatenate([x,y],1)
        print(x)

        # [
        # [1., 2., 3., 0., -1., -2.],
        # [4., 5., 6., -3., -4., -5.]
        # ]
        
        

stack
==============================

.. py:function:: pyvqnet.tensor.stack(QTensors: list, dim=0) 

    沿新维度 dim 堆叠输入的 QTensors ,返回一个新的 QTensor。

    :param QTensors: 包含输入 QTensor 。
    :param dim: 要堆叠的维度。 必须介于 0 和输入张量的最大维数之间。
    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape(R, C).astype(np.float32)
        t11 = QTensor(a)
        t22 = QTensor(a)
        t33 = QTensor(a)
        rlt1 = tensor.stack([t11,t22,t33],2)
        print(rlt1)
        
        # [
        # [[0., 0., 0.],
        #  [1., 1., 1.],
        #  [2., 2., 2.],
        #  [3., 3., 3.]],
        # [[4., 4., 4.],
        #  [5., 5., 5.],
        #  [6., 6., 6.],
        #  [7., 7., 7.]],
        # [[8., 8., 8.],
        #  [9., 9., 9.],
        #  [10., 10., 10.],
        #  [11., 11., 11.]]
        # ]
                

permute
==============================

.. py:function:: pyvqnet.tensor.permute(t: pyvqnet.tensor.QTensor, dim: list)

    根据输入的 dim 的顺序,改变t 的轴的顺序。

    :param t: 输入 QTensor 。
    :param dim: 维度的新顺序(整数列表),长度必须等于 ``t`` 的维数,且是 ``0..ndim-1`` 的一个排列;可以使用负数轴。不满足时抛出 ``ValueError``。
    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor, QTensor
        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape([2, 2, 3]).astype(np.float32)
        t = QTensor(a)
        rlt = tensor.permute(t,[2,0,1])
        print(rlt)

        print(tensor.permute(t, [-1, 0, 1]).shape)
        for dims in ([0, 1], [0, 0, 2], [0, 1, 3]):
            try:
                tensor.permute(t, dims)
            except ValueError as e:
                print("ValueError:", e)

        # [[[ 0., 3.],
        #   [ 6., 9.]],

        #  [[ 1., 4.],
        #   [ 7.,10.]],

        #  [[ 2., 5.],
        #   [ 8.,11.]]]
        # <QTensor [3, 2, 2] DEV_CPU float32>
        # [3, 2, 2]
        # ValueError: permute expected 3 dimensions, got 2
        # ValueError: permute dims must be a permutation of 0..2
        # ValueError: permute dims must be a permutation of 0..2

transpose
==============================

.. py:function:: pyvqnet.tensor.transpose(t: pyvqnet.tensor.QTensor, dim: list)

    根据输入的 dim 的顺序,改变t 的轴的顺序。如果 dims = None,则按顺序反转 t 的轴。

    :param t: 输入 QTensor 。
    :param dim: 维度的新顺序(整数列表)。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape([2,2,3]).astype(np.float32)
        t = QTensor(a)
        tt = tensor.transpose(t,[2,0,1])
        print(tt)

        # [
        # [[0., 3.],
        #  [6., 9.]],
        # [[1., 4.],
        #  [7., 10.]],
        # [[2., 5.],
        #  [8., 11.]]
        # ]
        

tile
==============================

.. py:function:: pyvqnet.tensor.tile(t: pyvqnet.tensor.QTensor, reps: list)


    通过按照 reps 给出的次数复制输入 QTensor 。
    
    如果 reps 的长度为 d,则结果 QTensor 的维度大小为 max(d, t.ndim)。如果 t.ndim < d,则通过从起始维度插入新轴,将 t 扩展为 d 维度。
    
    因此形状 (3,) 数组被提升为 (1, 3) 用于 2-D 复制,或形状 (1, 1, 3) 用于 3-D 复制。如果 t.ndim > d,reps 通过插入 1 扩展为 t.ndim。

    因此,对于形状为 (2, 3, 4, 5) 的 t,(4, 3) 的 reps 被视为 (1, 1, 4, 3)。

    :param t: 输入 QTensor 。
    :param reps: 每个维度的重复次数。
    :return: 一个新的 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        import numpy as np
        a = np.arange(6).reshape(2,3).astype(np.float32)
        A = QTensor(a)
        reps = [2,2]
        B = tensor.tile(A,reps)
        print(B)

        # [
        # [0., 1., 2., 0., 1., 2.],
        # [3., 4., 5., 3., 4., 5.],
        # [0., 1., 2., 0., 1., 2.],
        # [3., 4., 5., 3., 4., 5.]
        # ]
        

squeeze
==============================

.. py:function:: pyvqnet.tensor.squeeze(t: pyvqnet.tensor.QTensor, dim: Optional[int] = None)

    删除 dim 指定的轴,该轴的维度为1。如果 dim = None ,则将输入所有长度为1的维度删除。

    :param t: 输入 QTensor 。
    :param dim: 要压缩的轴,默认为None。 
    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.arange(6).reshape(1,6,1).astype(np.float32)
        A = QTensor(a)
        AA = tensor.squeeze(A,0)
        print(AA)

        # [
        # [0.],
        # [1.],
        # [2.],
        # [3.],
        # [4.],
        # [5.]
        # ]
        

unsqueeze
==============================

.. py:function:: pyvqnet.tensor.unsqueeze(t: pyvqnet.tensor.QTensor, dim: int = 0)

    在 dim 指定的维度上插入一个维度为的1的轴,返回一个新的 QTensor 。

    :param t: 输入 QTensor 。
    :param dim: 要插入维度的位置,默认为0。 
    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.arange(24).reshape(2,1,1,4,3).astype(np.float32)
        A = QTensor(a)
        AA = tensor.unsqueeze(A,1)
        print(AA)

        # [
        # [[[[[0., 1., 2.],
        #  [3., 4., 5.],
        #  [6., 7., 8.],
        #  [9., 10., 11.]]]]],
        # [[[[[12., 13., 14.],
        #  [15., 16., 17.],
        #  [18., 19., 20.],
        #  [21., 22., 23.]]]]]
        # ]
        

moveaxis
==============================
.. py:function:: pyvqnet.tensor.moveaxis(t, src, dst)

    将 `t` 的维度从 `src` 中的位置移动到 `dst` 中的位置。

    `t` 的其他未明确移动的维度保持其原始顺序,并出现在 `dst` 中未指定的位置。

    :param t: 输入 QTensor。
    :param src: (整数或整数元组)要移动的维度的原始位置。这些位置必须是唯一的。
    :param dst: (整数或整数元组)每个原始维度的目标位置。这些位置也必须是唯一的。

    :return: 新的QTensor

    Example::

        from pyvqnet import QTensor,tensor
        a = tensor.arange(0,24).reshape((2,3,4))
        b = tensor.moveaxis(a,(1, 2), (0, 1))
        print(b.shape)
        #[3, 4, 2]


swapaxis
==============================

.. py:function:: pyvqnet.tensor.swapaxis(t, dim1: int, dim2: int)

    交换输入 t 的 第 dim1 和 dim2 维度。

    :param t: 输入 QTensor 。
    :param dim1: 要交换的第一个轴。
    :param dim2:  要交换的第二个轴。
    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        import numpy as np
        a = np.arange(24).reshape(2,3,4).astype(np.float32)
        A = QTensor(a)
        AA = tensor.swapaxis(A, 2, 1)
        print(AA)

        # [
        # [[0., 4., 8.],
        #  [1., 5., 9.],
        #  [2., 6., 10.],
        #  [3., 7., 11.]],
        # [[12., 16., 20.],
        #  [13., 17., 21.],
        #  [14., 18., 22.],
        #  [15., 19., 23.]]
        # ]

masked_fill
==============================

.. py:function:: pyvqnet.tensor.masked_fill(t, mask, value)

    在 mask == 1 的位置,用值 value 填充输入。
    mask的形状必须与输入的 QTensor 的形状是可广播的。

    :param t: 输入 QTensor。
    :param mask: 掩码 QTensor,必须是kbool。
    :param value: 填充值。
    :return: 一个 QTensor。

    Example::

        from pyvqnet.tensor import tensor
        import numpy as np
        a = tensor.ones([2, 2, 2, 2])
        mask = np.random.randint(0, 2, size=4).reshape([2, 2])
        b = tensor.QTensor(mask==1)
        c = tensor.masked_fill(a, b, 13)
        print(c)
        # [
        # [[[1., 1.],  
        #  [13., 13.]],
        # [[1., 1.],   
        #  [13., 13.]]],
        # [[[1., 1.],
        #  [13., 13.]],
        # [[1., 1.],
        #  [13., 13.]]]
        # ]

flatten
==============================

.. py:function:: pyvqnet.tensor.flatten(t: pyvqnet.tensor.QTensor, start: int = 0, end: int = - 1)

    将输入 t 从 start 到 end 的连续维度展平。

    :param t: 输入 QTensor 。
    :param start: 展平开始的轴,默认 = 0,从第一个轴开始。
    :param end: 展平结束的轴,默认 = -1,以最后一个轴结束。
    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.flatten(t)
        print(x)

        # [1., 2., 3.]


reshape
==============================

.. py:function:: pyvqnet.tensor.reshape(t: pyvqnet.tensor.QTensor,shape)

    改变 QTensor 的形状,返回一个新的张量。

    :param t: 输入 QTensor 。
    :param shape: 新的形状。

    :return: 新形状的 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape(R, C).astype(np.float32)
        t = QTensor(a)
        reshape_t = tensor.reshape(t, [C, R])
        print(reshape_t)
        # [
        # [0., 1., 2.],
        # [3., 4., 5.],
        # [6., 7., 8.],
        # [9., 10., 11.]
        # ]

flip
==============================

.. py:function:: pyvqnet.tensor.flip(t, flip_dims)

    沿指定轴反转QTensor,返回一个新的张量。

    :param t: 输入 QTensor 。
    :param flip_dims: 需要翻转的轴或轴列表。

    :return: 新形状的 QTensor 。

    Example::

        from pyvqnet import tensor
        t = tensor.arange(1, 3 * 2 *2 * 2 + 1).reshape([3, 2, 2, 2])
        t.requires_grad = True
        y = tensor.flip(t, [0, -1])
        print(y)
        # [
        # [[[18., 17.], 
        #  [20., 19.]], 
        # [[22., 21.],  
        #  [24., 23.]]],
        # [[[10., 9.],  
        #  [12., 11.]], 
        # [[14., 13.],  
        #  [16., 15.]]],
        # [[[2., 1.],   
        #  [4., 3.]],   
        # [[6., 5.],    
        #  [8., 7.]]]   
        # ]

gather
==============================

.. py:function:: pyvqnet.tensor.gather(t, dim, index)

    沿由“dim”指定的轴收集值。

    对于 3-D 张量,输出由以下指定:

    .. math::

         out[i][j][k] = t[index[i][j][k]][j][k] , 如果 dim == 0 \\

         out[i][j][k] = t[i][index[i][j][k]][k] , 如果 dim == 1 \\

         out[i][j][k] = t[i][j][index[i][j][k]] , 如果 dim == 2 \\

    :param t: 输入 QTensor。
    :param dim: 聚集轴。
    :param index: 索引QTensor,应该与输入具有相同的维度大小。

    :return: 聚集的结果

    Example::

        from pyvqnet.tensor import gather,QTensor,tensor
        import numpy as np
        np.random.seed(25)
        npx = np.random.randn( 3, 4,6)
        npindex = np.array([2,3,1,2,1,2,3,0,2,3,1,2,3,2,0,1]).reshape([2,2,4]).astype(np.int64)

        x1 = QTensor(npx)
        indices1 =  QTensor(npindex)
        x1.requires_grad = True
        y1 = gather(x1,1,indices1)
        y1.backward(tensor.arange(0,y1.numel()).reshape(y1.shape))

        print(y1)
        # [
        # [[2.1523438, -0.4196777, -2.0527344, -1.2460938],
        #  [-0.6201172, -1.3349609, 2.2949219, -0.5913086]],
        # [[0.2170410, -0.7055664, 1.6074219, -1.9394531],
        #  [0.2430420, -0.6333008, 0.5332031, 0.3881836]]
        # ]


scatter
==============================

.. py:function:: pyvqnet.tensor.scatter(input, dim, indices, src)


    将张量 src 中的所有值写入 indices 张量中指定的索引处的 input 中。

    对于 3-D 张量,输出由以下指定:

    .. math::

        input[indices[i][j][k]][j][k] = src[i][j][k] , 如果 dim == 0 \\
        input[i][indices[i][j][k]][k] = src[i][j][k] , 如果 dim == 1 \\
        input[i][j][indices[i][j][k]] = src[i][j][k] , 如果 dim == 2 \\

    :param input: 输入QTensor。
    :param dim: 散点轴。
    :param indices: 索引QTensor,应该和输入有相同的维度大小。
    :param src: 要散布的源张量。

    Example::

        from pyvqnet.tensor import scatter, QTensor
        import numpy as np
        np.random.seed(25)
        npx = np.random.randn(3, 2, 4, 2)
        npindex = np.array([2, 3, 1, 2, 1, 2, 3, 0, 2, 3, 1, 2, 3, 2, 0,
                            1]).reshape([2, 2, 4, 1]).astype(np.int64)
        x1 = QTensor(npx)
        npsrc = QTensor(np.full_like(npindex, 200), dtype=x1.dtype)
        npsrc.requires_grad = True
        indices1 = QTensor(npindex)
        y1 = scatter(x1, 2, indices1, npsrc)
        print(y1)

        # [[[[  0.2282731   1.0268903]
        #    [200.         -0.5911815]
        #    [200.         -0.2223257]
        #    [200.          1.8379046]]

        #   [[200.          0.8685831]
        #    [200.         -0.2323119]
        #    [200.         -1.3346615]
        #    [200.         -1.2460893]]]


        #  [[[  1.2022723  -1.0499416]
        #    [200.         -0.4196777]
        #    [200.         -2.5944874]
        #    [200.          0.6808889]]

        #   [[200.         -1.9762536]
        #    [200.         -0.2908697]
        #    [200.          1.9826261]
        #    [200.         -1.839905 ]]]


        #  [[[  1.6076708   0.3882919]
        #    [  0.3997321   0.4054766]
        #    [  0.2170018  -0.6334391]
        #    [  0.2466215  -1.9395455]]

        #   [[  0.1140596  -1.8853414]
        #    [  0.2430805  -0.7054807]
        #    [  0.3646276  -0.5029522]
        #    [ -0.2257515  -0.5655377]]]]

broadcast_to
==============================

.. py:function:: pyvqnet.tensor.broadcast_to(t, ref)

    受到某些约束,数组 t 被“广播”到参考形状,以便它们具有兼容的形状。

    https://numpy.org/doc/stable/user/basics.broadcasting.html

    :param t: 输入QTensor
    :param ref: 参考形状。
    
    :return: 新广播的 t 的 QTensor。

    Example::

        from pyvqnet.tensor.tensor import QTensor
        from pyvqnet.tensor import *
        ref = [2,3,4]
        a = ones([4])
        b = tensor.broadcast_to(a,ref)
        print(b.shape)
        #[2, 3, 4]



实用函数
************


to_tensor
==============================

.. py:function:: pyvqnet.tensor.to_tensor(x)

    将输入数值或 numpy.ndarray 等转换为 QTensor 。

    :param x: 整数、浮点数、布尔数、复数、或 numpy.ndarray
    :return: 输出 QTensor

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.to_tensor(10.0)
        print(t)

        # [10.]
        

pad_sequence
==============================

.. py:function:: pyvqnet.tensor.pad_sequence(qtensor_list, batch_first=False, padding_value=0)

    用 ``padding_value`` 填充可变长度张量列表。 ``pad_sequence`` 沿新维度堆叠张量列表,并将它们填充到相等的长度。
    输入是列表大小为 ``L x *`` 的序列。 L 是可变长度。

    :param qtensor_list: `list[QTensor]`- 可变长度序列列表。
    :param batch_first: 'bool' - 如果为真,输出将是 ``批大小 x 最长序列长度 x *`` ,否则为 ``最长序列长度 x 批大小 x *`` 。 默认值: False。
    :param padding_value: 'float' - 填充值。 默认值:0。

    :return:
        如果 batch_first 为 ``False``,则张量大小为 ``批大小 x 最长序列长度 x *``。
        否则张量的大小为 ``最长序列长度 x 批大小 x *`` 。

    Example::

        from pyvqnet.tensor import tensor
        a = tensor.ones([4, 2,3])
        b = tensor.ones([1, 2,3])
        c = tensor.ones([2, 2,3])
        a.requires_grad = True
        b.requires_grad = True
        c.requires_grad = True
        y = tensor.pad_sequence([a, b, c], True)

        print(y)
        # [
        # [[[1., 1., 1.],
        #  [1., 1., 1.]],
        # [[1., 1., 1.],
        #  [1., 1., 1.]],
        # [[1., 1., 1.],
        #  [1., 1., 1.]],
        # [[1., 1., 1.],
        #  [1., 1., 1.]]],
        # [[[1., 1., 1.],
        #  [1., 1., 1.]],
        # [[0., 0., 0.],
        #  [0., 0., 0.]],
        # [[0., 0., 0.],
        #  [0., 0., 0.]],
        # [[0., 0., 0.],
        #  [0., 0., 0.]]],
        # [[[1., 1., 1.],
        #  [1., 1., 1.]],
        # [[1., 1., 1.],
        #  [1., 1., 1.]],
        # [[0., 0., 0.],
        #  [0., 0., 0.]],
        # [[0., 0., 0.],
        #  [0., 0., 0.]]]
        # ]


pad_packed_sequence
==============================

.. py:function:: pyvqnet.tensor.pad_packed_sequence(sequence, batch_first=False, padding_value=0, total_length=None)

    填充一批打包的可变长度序列。它是 `pack_pad_sequence` 的逆操作。
    当  ``batch_first`` 是 True,它将返回  ``B x T x *`` 形状的张量,否则返回  ``T x B x *``。
    其中 `T` 为序列最长长度, `B` 为批处理大小。



    :param sequence: 'QTensor' - 待处理数据。
    :param batch_first: 'bool' - 如果为 ``True`` ,批处理将是输入的第一维。 默认值:False。
    :param padding_value: 'bool' - 填充值。默认:0。
    :param total_length: 'bool' - 如果不是 ``None`` ,输出将被填充到长度 :attr:`total_length`。 默认值:None。
    :return:
        包含填充序列的张量元组,以及批次中每个序列的长度列表。批次元素将按照最初的顺序重新排序。

    Example::

        from pyvqnet.tensor import tensor
        a = tensor.ones([4, 2,3])
        b = tensor.ones([2, 2,3])
        c = tensor.ones([1, 2,3])
        a.requires_grad = True
        b.requires_grad = True
        c.requires_grad = True
        y = tensor.pad_sequence([a, b, c], True)
        seq_len = [4, 2, 1]
        data = tensor.pack_pad_sequence(y,
                                seq_len,
                                batch_first=True,
                                enforce_sorted=True)

        seq_unpacked, lens_unpacked = tensor.pad_packed_sequence(data, batch_first=True)
        print(seq_unpacked)
        # [[[[1. 1. 1.]
        #    [1. 1. 1.]]

        #   [[1. 1. 1.]
        #    [1. 1. 1.]]

        #   [[1. 1. 1.]
        #    [1. 1. 1.]]

        #   [[1. 1. 1.]
        #    [1. 1. 1.]]]


        #  [[[1. 1. 1.]
        #    [1. 1. 1.]]

        #   [[1. 1. 1.]
        #    [1. 1. 1.]]

        #   [[0. 0. 0.]
        #    [0. 0. 0.]]

        #   [[0. 0. 0.]
        #    [0. 0. 0.]]]


        #  [[[1. 1. 1.]
        #    [1. 1. 1.]]

        #   [[0. 0. 0.]
        #    [0. 0. 0.]]

        #   [[0. 0. 0.]
        #    [0. 0. 0.]]

        #   [[0. 0. 0.]
        #    [0. 0. 0.]]]]
        print(lens_unpacked)
        # [4, 2, 1]


pack_pad_sequence
==============================

.. py:function:: pyvqnet.tensor.pack_pad_sequence(input, lengths, batch_first=False, enforce_sorted=True)

    打包一个包含可变长度填充序列的张量。
    如果 batch_first 是 True, `input` 的形状应该为 [批大小,长度,*],否则形状 [长度,批大小,*]。

    对于未排序的序列,使用 ``enforce_sorted`` 是 False。 如果 :attr:`enforce_sorted` 是 ``True``,序列应该按长度降序排列。

    :param input: 'QTensor' - 填充的可变长度序列。
    :param lengths: 'list' - 每个批次的序列长度。
    :param batch_first: 'bool' - 如果 ``True``,则输入预期为 ``B x T x *``
        格式,默认:False。
    :param enforce_sorted: 'bool' - 如果 ``True``,输入应该是
        包含按长度降序排列的序列。 如果 ``False``,输入将无条件排序。 默认值:True。

    :return: 一个 :class:`PackedSequence` 对象。

    Example::

        from pyvqnet.tensor import tensor
        a = tensor.ones([4, 2,3])
        c = tensor.ones([1, 2,3])
        b = tensor.ones([2, 2,3])
        a.requires_grad = True
        b.requires_grad = True
        c.requires_grad = True
        y = tensor.pad_sequence([a, b, c], True)
        seq_len = [4, 2, 1]
        data = tensor.pack_pad_sequence(y,
                                seq_len,
                                batch_first=True,
                                enforce_sorted=False)
        print(data.data)

        # [[[1. 1. 1.]
        #   [1. 1. 1.]]

        #  [[1. 1. 1.]
        #   [1. 1. 1.]]

        #  [[1. 1. 1.]
        #   [1. 1. 1.]]

        #  [[1. 1. 1.]
        #   [1. 1. 1.]]

        #  [[1. 1. 1.]
        #   [1. 1. 1.]]

        #  [[1. 1. 1.]
        #   [1. 1. 1.]]

        #  [[1. 1. 1.]
        #   [1. 1. 1.]]]

        print(data.batch_sizes)
        # [3, 2, 1, 1]

functional_conv2d
==============================
.. py:function:: pyvqnet.nn.common.functional.functional_conv2d(x, weight, bias, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1)

    对由多个输入平面组成的输入图像进行二维卷积。

    :param x: 4维输入张量。
    :param weight: 4维核张量。


    :param stride: `tuple` - 步长，默认为 (1, 1)
    :param padding: 填充，控制输入的填充量。它可以是字符串 {‘valid’, ‘same’} 或一个整数元组，用于指定应用于输入的隐式填充量，默认为(0,0)
    :param dilation: `tuple` - 核元素之间的间距。默认值：(0,0)
    :param groups: `int` - 组数。默认值：1

    :return: qtensor


    Example::

        from pyvqnet.nn.common.functional import functional_conv2d
        from pyvqnet.tensor import arange,ones
        from pyvqnet import kfloat32
        from pyvqnet.nn import Module,Parameter


        class TM(Module):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.w = ones([5,4,2,2])
                self.w.requires_grad = True
                self.b = ones([5,])
                self.b.requires_grad = True

            def forward(self,x):
                weight, bias,   = self.w, self.b 
                return functional_conv2d(x, weight, bias )


        x = arange(0,7*4*12*12,dtype=kfloat32).reshape([7,4,12,12])
        l = TM()
        y = l(x)

        y.backward( )

no_grad
==============================
    
.. py:function:: pyvqnet.no_grad()

    禁用前向计算时记录反向传播节点。

    Example::

        import pyvqnet.tensor as tensor
        from pyvqnet import no_grad

        with no_grad():
            x = tensor.QTensor([1.0, 2.0, 3.0],requires_grad=True)
            y = tensor.tan(x)
            y.backward()
        """
        RuntimeError: The output tensor does not require gradients (output.requires_grad == False). This may occur if you used a non-autograd function in the forward pass. To enable gradient computation, ensure that all operations are performed on tensors with requires_grad=True, or use autograd-compatible functions.
        """