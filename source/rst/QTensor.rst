.. _qtensor_api:

###############
 QTensor 模块
###############

VQNet量子机器学习所使用的数据结构QTensor的python接口介绍。QTensor支持常用的多维张量的操作,例如创建函数,数学函数,逻辑函数,矩阵变换等。

**********************
 QTensor's 函数与属性
**********************

.. py:class:: pyvqnet.tensor.tensor.QTensor(data, requires_grad=False, nodes=None, device=0, dtype=None, name="")

   具有动态计算图构造和自动微分的张量。

   :param data: 输入数据,可以是 _core.Tensor 或numpy 数组。
   :param requires_grad: 是否应该跟踪张量的梯度,默认为 False。
   :param nodes: 计算图中的后继者列表,默认为无。
   :param device: 储存在哪个设备上,默认: pyvqnet.DEV_CPU,在CPU上。
   :param dtype: 参数的数据类型,defaults:None,使用默认数据类型:kfloat32,代表32位浮点数。
   :param name: QTensor的名字,default:""。
   :return: 输出 QTensor。

   .. note::

           QTensor 内部数据类型dtype支持kbool,kuint8,kint8,kint16,kint32,kint64,kfloat32,kfloat64,kcomplex64,kcomplex128.
           分别对应C++的 bool,uint8_t,int8_t,int16_t,int32_t,int64_t,float,double,complex<float>,complex<double>.

   Example::

       from pyvqnet.tensor import QTensor
       from pyvqnet.dtype import *
       import numpy as np

       t1 = QTensor(np.ones([2,3]))
       t2 =  QTensor([2,3,4j,5])
       t3 =  QTensor([[[2,3,4,5],[2,3,4,5]]],dtype=kbool)
       print(t1)
       print(t2)
       print(t3)



   .. py:attribute:: ndim

       返回张量的维度的个数。

       :return: 张量的维度的个数。

       Example::

           from pyvqnet.tensor import QTensor

           a = QTensor([2, 3, 4, 5], requires_grad=True)
           print(a.ndim)



   .. py:attribute:: shape

       返回张量的维度

       :return: 一个列表存有张量的维度

       Example::

           from pyvqnet.tensor import QTensor

           a = QTensor([2, 3, 4, 5], requires_grad=True)
           print(a.shape)



   .. py:attribute:: size

       返回张量的元素个数。

       :return: 张量的元素个数。

       Example::

           from pyvqnet.tensor import QTensor

           a = QTensor([2, 3, 4, 5], requires_grad=True)
           print(a.size)





   .. py:method:: numel

       返回张量的元素个数。

       :return: 张量的元素个数。

       Example::

           from pyvqnet.tensor import QTensor

           a = QTensor([2, 3, 4, 5], requires_grad=True)
           print(a.numel())





   .. py:attribute:: dtype

       返回张量的数据类型。

       QTensor 内部数据类型dtype支持kbool = 0, kuint8 = 1, kint8 = 2,kint16 = 3,kint32 = 4,kint64 = 5,
       kfloat32 = 6, kfloat64 = 7, kcomplex64 = 8, kcomplex128 = 9 .

       :return: 张量的数据类型。

       Example::

           from pyvqnet.tensor import QTensor

           a = QTensor([2, 3, 4, 5])
           print(a.dtype)



   .. py:attribute:: is_dense

       是否是稠密张量。

       :return: 当该数据是稠密的时候, 返回1;否则返回 0。

       Example::

           from pyvqnet.tensor import QTensor

           a = QTensor([2, 3, 4, 5])
           print(a.is_dense)



   .. py:attribute:: is_csr

       是否是Compressed Sparse Row格式的稀疏2维度矩阵。

       :return: 当该数据是CSR格式的稀疏张量时候, 返回1;否则返回 0。

       Example::

           from pyvqnet.tensor import QTensor,dense_to_csr

           a = QTensor([[2, 3, 4, 5]])
           b = dense_to_csr(a)
           print(b.is_csr)



   .. py:attribute:: is_contiguous

       是否是contiguous的多维数组。

       :return: 如果是contiguous, 返回True, 否则返回False。

       Example::

           from pyvqnet.tensor import QTensor

           a = QTensor([[2, 3, 4, 5],[2, 3, 4, 5]])
           b = a.is_contiguous
           print(b)

           c= a.permute((1,0))
           print(c.is_contiguous)




   .. py:method:: csr_members()

       返回Compressed Sparse Row格式的稀疏2维度矩阵的row_idx,col_idx 以及非0数值data,3个1维QTensor。具体含义见 https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)。

       :return:

           返回列表, 其中第一个元素为row_idx,shape为[矩阵行数+1],第2个元素为col_idx,shape为[非0元素数], 第3个元素为data,shape为[非0元素数]

       Example::

           from pyvqnet.tensor import QTensor,dense_to_csr

           a = QTensor([[2, 3, 4, 5]])
           b = dense_to_csr(a)
           print(b.csr_members())




   .. py:method:: zero_grad()

       将张量的梯度设置为零。将在优化过程中被优化器使用。

       :return: 无。

       Example::

           from pyvqnet.tensor import QTensor
           t3 = QTensor([2, 3, 4, 5], requires_grad=True)
           t3.zero_grad()
           print(t3.grad)




   .. py:method:: backward(grad=None)

       利用反向传播算法, 计算当前张量所在的计算图中的所有需计算梯度的张量的梯度。

       :param grad: 输入的梯度,如果grad 是None 则输入全1的梯度，默认:None。
       :return: 无

       Example::

           from pyvqnet.tensor import QTensor

           target = QTensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0.2]], requires_grad=True)
           y = 2*target + 3
           y.backward()
           print(target.grad)



   .. py:method:: to_numpy()

       将张量的数据拷贝到一个numpy.ndarray里面。

       :return: 一个新的 numpy.ndarray 包含 QTensor 数据

       Example::

           from pyvqnet.tensor import tensor
           from pyvqnet.tensor import QTensor
           t3 = QTensor([2, 3, 4, 5], requires_grad=True)
           t4 = t3.to_numpy()
           print(t4)



   .. py:method:: item()

       从只包含单个元素的 QTensor 返回唯一的元素。

       :return: 元素值

       Example::

           from pyvqnet.tensor import tensor

           t = tensor.ones([1])
           print(t.item())



   .. py:method:: contiguous()

       返回当前QTensor的contiguous形式 ,如果已经是contiguous, 则返回自身。

       :return: 返回当前QTensor的contiguous形式 ,如果已经是contiguous, 则返回自身。

       Example::

           from pyvqnet.tensor import tensor

           t = tensor.ones([1])





   .. py:method:: argmax(*kargs)

       返回输入 QTensor 中所有元素的最大值的索引, 或返回 QTensor 按某一维度的最大值的索引。

       :param dim: 计算argmax的轴, 只接受单个维度。 如果 dim == None, 则返回输入张量中所有元素的最大值的索引。有效的 dim 范围是 [-R, R), 其中 R 是输入的 ndim。 当 dim < 0 时, 它的工作方式与 dim + R 相同。
       :param keepdims: 输出 QTensor 是否保留了最大值索引操作的轴, 默认是False。

       :return: 输入 QTensor 中最大值的索引。

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




   .. py:method:: argmin(*kargs)

       返回输入 QTensor 中所有元素的最小值的索引, 或返回 QTensor 按某一维度的最小值的索引。

       :param dim: 计算argmax的轴, 只接受单个维度。 如果 dim == None, 则返回输入张量中所有元素的最小值的索引。有效的 dim 范围是 [-R, R), 其中 R 是输入的 ndim。 当 dim < 0 时, 它的工作方式与 dim + R 相同。
       :param keepdims: 输出 QTensor 是否保留了最小值索引操作的轴, 默认是False。

       :return: 输入 QTensor 中最小值的索引。

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


   .. py:method:: fill_(v)

       为当前张量填充特定值, 该函数改变原张量的内部数据。

       :param v: 填充值。

       :return: 无。

       Example::

           from pyvqnet.tensor import tensor
           from pyvqnet.tensor import QTensor
           shape = [2, 3]
           value = 42
           t = tensor.zeros(shape)
           t.fill_(value)
           print(t)



   .. py:method:: all()

       判断张量内数据是否全为全零。

       :return: 返回True, 如果全为非0;否则返回False。

       Example::

           from pyvqnet.tensor import tensor

           shape = [2, 3]
           t = tensor.zeros(shape)
           t.fill_(1.0)
           flag = t.all()
           print(flag)



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




   .. py:method:: fill_rand_binary_(v=0.5)

       用从二项分布中随机采样的值填充 QTensor 。

       如果二项分布后随机生成的数据大于二值化阈值 v , 则设置 QTensor 对应位置的元素值为1, 否则为0。

       :param v: 二值化阈值, 默认0.5。

       :return: 无。

       Example::

           from pyvqnet.tensor import tensor
           from pyvqnet.tensor import QTensor
           import numpy as np
           a = np.arange(6).reshape(2, 3).astype(np.float32)
           t = QTensor(a)
           t.fill_rand_binary_(2)
           print(t)



   .. py:method:: fill_rand_signed_uniform_(v=1)

       用从有符号均匀分布中随机采样的值填充 QTensor 。用缩放因子 v 对生成的随机采样的值进行缩放。

       :param v: 缩放因子, 默认1。

       :return: 无。

       Example::

           from pyvqnet.tensor import tensor
           from pyvqnet.tensor import QTensor
           import numpy as np
           a = np.arange(6).reshape(2, 3).astype(np.float32)
           t = QTensor(a)
           value = 42

           t.fill_rand_signed_uniform_(value)
           print(t)



   .. py:method:: fill_rand_uniform_(v=1)

       用从均匀分布中随机采样的值填充 QTensor 。用缩放因子 v 对生成的随机采样的值进行缩放。

       :param v: 缩放因子, 默认1。

       :return: 无。

       Example::

           from pyvqnet.tensor import tensor
           from pyvqnet.tensor import QTensor
           import numpy as np
           a = np.arange(6).reshape(2, 3).astype(np.float32)
           t = QTensor(a)
           value = 42
           t.fill_rand_uniform_(value)
           print(t)



   .. py:method:: fill_rand_normal_(m=0, s=1, fast_math=True)

       生成均值为 m 和方差 s 产生正态分布元素, 并填充到张量中。

       :param m: 均值, 默认0。
       :param s: 方差, 默认1。
       :param fast_math: 是否使用快速方法产生高斯分布, 默认True。

       :return: 无。

       Example::

           from pyvqnet.tensor import tensor
           from pyvqnet.tensor import QTensor
           import numpy as np
           a = np.arange(6).reshape(2, 3).astype(np.float32)
           t = QTensor(a)
           t.fill_rand_normal_(2, 10, True)
           print(t)



   .. py:method:: transpose(new_dims=None)

       反转张量的轴。如果 new_dims = None, 则反转所有轴。

       :param new_dims: 列表形式储存的新的轴顺序,默认: None, 则反转所有轴。

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



   .. py:method:: reshape(new_shape)

       改变 QTensor 的形状, 返回一个新的张量。

       :param new_shape: 新的形状。

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



   .. py:method:: __getitem__(item )

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

           print(aaa[3, 4, 1])

           print(aaa[:, 2, :])

           print(aaa[2])

           print(aaa[0:2, ::3, 2:])

           a = tensor.ones([2, 2])
           b = QTensor([[1, 1], [0, 1]])
           b = b > 0
           c = a[b]
           print(c)

           tt = tensor.arange(1, 56 * 2 * 4 * 4 + 1).reshape([2, 8, 4, 7, 4])
           tt.requires_grad = True
           index_sample1 = tensor.arange(0, 3).reshape([3, 1])
           index_sample2 = QTensor([0, 1, 0, 2, 3, 2, 2, 3, 3]).reshape([3, 3])
           gg = tt[:, index_sample1, 3:, index_sample2, 2:]
           print(gg)




   .. py:method:: __setitem__(key, value)

       支持对 QTensor 使用 切片索引, 下标, 或使用 QTensor 作为高级索引修改输入。该操作对输入原地进行修改 。

       通过冒号 ``:``  分隔切片参数 start:stop:step 来进行切片操作, 其中 start、stop、step 均可缺省。

       针对1-D QTensor, 则仅有单个轴上的索引或切片。

       针对2-D及以上的 QTensor , 则会有多个轴上的索引或切片。

       使用 QTensor 作为 索引, 则进行高级索引, 请参考numpy中 `高级索引 <https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html>`_ 部分。

       若作为索引的 QTensor 为逻辑运算的结果, 则进行 布尔数组索引。

       .. note::

           a[3][4][1] 形式的索引暂不支持, 使用 a[3,4,1] 形式代替。


       :param key: 以 pyslice , 整数, QTensor 构成切片索引。
       :param value: 填充到 `key` 所在位置的数值或者QTensor 
       :return: 无。

       Example::

           from pyvqnet.tensor import tensor
           aaa = tensor.arange(1, 61)
           aaa = aaa.reshape([4, 5, 3])
           vqnet_a2 = aaa[3, 4, 1]
           aaa[3, 4, 1] = tensor.arange(10001,
                                           10001 + vqnet_a2.size).reshape(vqnet_a2.shape)
           print(aaa)

           aaa = tensor.arange(1, 61)
           aaa = aaa.reshape([4, 5, 3])
           vqnet_a3 = aaa[:, 2, :]
           aaa[:, 2, :] = tensor.arange(10001,
                                           10001 + vqnet_a3.size).reshape(vqnet_a3.shape)
           print(aaa)

           aaa = tensor.arange(1, 61)
           aaa = aaa.reshape([4, 5, 3])
           vqnet_a4 = aaa[2, :]
           aaa[2, :] = tensor.arange(10001,
                                       10001 + vqnet_a4.size).reshape(vqnet_a4.shape)
           print(aaa)

           aaa = tensor.arange(1, 61)
           aaa = aaa.reshape([4, 5, 3])
           vqnet_a5 = aaa[0:2, ::2, 1:2]
           aaa[0:2, ::2,
               1:2] = tensor.arange(10001,
                                       10001 + vqnet_a5.size).reshape(vqnet_a5.shape)
           print(aaa)

           a = tensor.ones([2, 2])
           b = tensor.QTensor([[1, 1], [0, 1]])
           b = b > 0
           x = tensor.QTensor([1001, 2001, 3001])

           a[b] = x
           print(a)




   .. py:method:: isCPU()

       该 QTensor 的数据是否存储在CPU主机内存上。

       :return: 该 QTensor 的数据是否存储在CPU主机内存上。

       Examples::

           from pyvqnet.tensor import QTensor
           a = QTensor([2])
           a = a.isCPU()
           print(a)


   .. py:method:: astype(dtype)

       将当前QTensor转化为对应数据类型dtype, 如果dtype相同则返回自身。

       :param dtype: 目标数据类型dtype。

       :return:  QTensor。

       Examples::

           from pyvqnet.tensor import QTensor
           from pyvqnet import kcomplex128
           a = QTensor([2])
           a = a.astype(kcomplex128)
           print(a)

***********
 创建函数
***********

.. _ones:

ones
====

.. py:function:: pyvqnet.tensor.ones(shape,device=pyvqnet.DEV_CPU,dtype-None)

   创建元素全一的 QTensor 。

   :param shape: 数据的形状。
   :param device: 储存在哪个设备上,默认: pyvqnet.DEV_CPU,在CPU上。
   :param dtype: 参数的数据类型,defaults:None,使用默认数据类型:kfloat32,代表32位浮点数。

   :return: 返回新的 QTensor 。

   Example::

       from pyvqnet.tensor import tensor

       x = tensor.ones([2, 3])
       print(x)

ones_like
=========

.. py:function:: pyvqnet.tensor.ones_like(t: pyvqnet.tensor.QTensor,device=pyvqnet.DEV_CPU,dtype=None)

   创建元素全一的 QTensor ,形状和输入的 QTensor 一样。

   :param t: 输入 QTensor 。
   :param device: 储存在哪个设备上,默认: pyvqnet.DEV_CPU,在CPU上。
   :param dtype: 参数的数据类型,defaults:None,跟输入的dtype一样。

   :return: 新的全一  QTensor 。

   Example::

       from pyvqnet.tensor import tensor
       from pyvqnet.tensor import QTensor
       t = QTensor([1, 2, 3])
       x = tensor.ones_like(t)
       print(x)

full
====

.. py:function:: pyvqnet.tensor.full(shape, value, device=pyvqnet.DEV_CPU, dtype=None)

   创建一个指定形状的 QTensor 并用特定值填充它。

   :param shape: 要创建的张量形状。
   :param value: 填充的值。
   :param device: 储存在哪个设备上,默认: pyvqnet.DEV_CPU,在CPU上。
   :param dtype: 参数的数据类型,defaults:None,使用默认数据类型:kfloat32,代表32位浮点数。

   :return: 输出新 QTensor 。

   Example::

       from pyvqnet.tensor import tensor
       from pyvqnet.tensor import QTensor
       shape = [2, 3]
       value = 42
       t = tensor.full(shape, value)
       print(t)

full_like
=========

.. py:function:: pyvqnet.tensor.full_like(t, value, device=pyvqnet.DEV_CPU,dtype=None)

   创建一个形状和输入一样的 QTensor,所有元素填充 value 。

   :param t: 输入 QTensor 。
   :param value: 填充 QTensor 的值。
   :param device: 储存在哪个设备上,默认: pyvqnet.DEV_CPU,在CPU上。
   :param dtype: 参数的数据类型,defaults:None,跟输入的dtype一样。

   :return: 输出 QTensor。

   Example::

       from pyvqnet.tensor import tensor
       from pyvqnet.tensor import QTensor
       a = tensor.randu([3,5])
       value = 42
       t = tensor.full_like(a, value)
       print(t)

zeros
=====

.. py:function:: pyvqnet.tensor.zeros(shape, device=pyvqnet.DEV_CPU,dtype=None)

   创建输入形状大小的全零 QTensor 。

   :param shape: 输入形状。
   :param device: 储存在哪个设备上,默认: pyvqnet.DEV_CPU,在CPU上。
   :param dtype: 参数的数据类型,defaults:None,使用默认数据类型:kfloat32,代表32位浮点数。

   :return: 输出 QTensor 。

   Example::

       from pyvqnet.tensor import tensor
       from pyvqnet.tensor import QTensor
       t = tensor.zeros([2, 3, 4])
       print(t)

zeros_like
==========

.. py:function:: pyvqnet.tensor.zeros_like(t: pyvqnet.tensor.QTensor, device=pyvqnet.DEV_CPU,dtype=None)

   创建一个形状和输入一样的 QTensor,所有元素为0 。

   :param t: 输入参考 QTensor 。
   :param device: 储存在哪个设备上,默认: pyvqnet.DEV_CPU,在CPU上。
   :param dtype: 参数的数据类型,defaults:None,跟输入的dtype一样。

   :return: 输出 QTensor 。

   Example::

       from pyvqnet.tensor import tensor
       from pyvqnet.tensor import QTensor
       t = QTensor([1, 2, 3])
       x = tensor.zeros_like(t)
       print(x)

arange
======

.. py:function:: pyvqnet.tensor.arange(start, end, step=1, device=pyvqnet.DEV_CPU,dtype=None,requires_grad=False)

   创建一个在给定间隔内具有均匀间隔值的一维 QTensor 。

   :param start: 间隔开始。
   :param end: 间隔结束。
   :param step: 值之间的间距,默认为1。
   :param device: 要使用的设备,默认 = pyvqnet.DEV_CPU,使用CPU设备。
   :param dtype: 参数的数据类型,defaults:None,使用默认数据类型:kfloat32,代表32位浮点数。
   :param requires_grad: 是否计算梯度,默认为False。

   :return: 输出 QTensor 。

   Example::

       from pyvqnet.tensor import tensor
       from pyvqnet.tensor import QTensor
       t = tensor.arange(2, 30, 4)
       print(t)

linspace
========

.. py:function:: pyvqnet.tensor.linspace(start, end, num, device=pyvqnet.DEV_CPU,dtype=None,requires_grad= False)

   创建一维 QTensor ,其中的元素为区间 start 和 end 上均匀间隔的共 num 个值。

   :param start: 间隔开始。
   :param end: 间隔结束。
   :param num: 间隔的个数。
   :param device: 要使用的设备,默认 = pyvqnet.DEV_CPU ,使用CPU设备。
   :param dtype: 参数的数据类型,defaults:None,使用默认数据类型:kfloat32,代表32位浮点数。
   :param requires_grad: 是否计算梯度,默认为False。

   :return: 输出 QTensor 。

   Example::

       from pyvqnet.tensor import tensor
       from pyvqnet.tensor import QTensor
       start, stop, num = -2.5, 10, 10
       t = tensor.linspace(start, stop, num)
       print(t)

logspace
========

.. py:function:: pyvqnet.tensor.logspace(start, end, num, base, device=pyvqnet.DEV_CPU,dtype=None, requires_grad)

   在对数刻度上创建具有均匀间隔值的一维 QTensor。

   :param start: ``base ** start`` 是起始值
   :param end: ``base ** end`` 是序列的最终值
   :param num: 要生成的样本数
   :param base: 对数刻度的基数
   :param device: 要使用的设备,默认 = pyvqnet.DEV_CPU ,使用CPU设备。
   :param dtype: 参数的数据类型,defaults:None,使用默认数据类型:kfloat32,代表32位浮点数。
   :param requires_grad: 是否计算梯度,默认为False。

   :return: 输出 QTensor 。

   Example::

       from pyvqnet.tensor import tensor
       from pyvqnet.tensor import QTensor
       start, stop, steps, base = 0.1, 1.0, 5, 10.0
       t = tensor.logspace(start, stop, steps, base)
       print(t)

eye
===

.. py:function:: pyvqnet.tensor.eye(size, offset: int = 0, device=pyvqnet.DEV_CPU,dtype=None)

   创建一个 size x size 的 QTensor,对角线上为 1,其他地方为 0。

   :param size: 要创建的(正方形)QTensor 的大小。
   :param offset: 对角线的索引:0(默认)表示主对角线,正值表示上对角线,负值表示下对角线。
   :param device: 要使用的设备,默认 =pyvqnet.DEV_CPU ,使用CPU设备。
   :param dtype: 参数的数据类型,defaults:None,使用默认数据类型:kfloat32,代表32位浮点数。

   :return: 输出 QTensor 。

   Example::

       from pyvqnet.tensor import tensor
       from pyvqnet.tensor import QTensor
       size = 3
       t = tensor.eye(size)
       print(t)

diagonal
========

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

diag
====

.. py:function:: pyvqnet.tensor.diag(t, k: int = 0)

   构造对角矩阵。

   输入一个 2-D QTensor,则返回一个1D的新张量,包含
   选定对角线中的元素。
   输入一个 1-D QTensor,则返回一个2D新张量,其选定对角线元素为输入值,其余为0

   :param t: 输入 QTensor。
   :param k: 偏移量(主对角线为 0, 正数为向上偏移, 负数为向下偏移), 默认为0。

   :return: 输出 QTensor。

   Example::

       from pyvqnet.tensor import tensor
       from pyvqnet.tensor import QTensor
       import numpy as np
       a = np.arange(16).reshape(4, 4).astype(np.float32)
       t = QTensor(a)
       for k in range(-3, 4):
           u = tensor.diag(t,k=k)
           print(u)

randu
=====

.. py:function:: pyvqnet.tensor.randu(shape, min=0.0,max=1.0, device=pyvqnet.DEV_CPU, dtype=None, requires_grad=False)

   创建一个具有均匀分布随机值的 QTensor 。

   :param shape: 要创建的 QTensor 的形状。
   :param min: 分布的下限,默认: 0。
   :param max: 分布的上线,默认: 1。
   :param device: 要使用的设备,默认 =pyvqnet.DEV_CPU ,使用CPU设备。
   :param dtype: 参数的数据类型,defaults:None,使用默认数据类型:kfloat32,代表32位浮点数。
   :param requires_grad: 是否计算梯度,默认为False。

   :return: 输出 QTensor 。

   Example::

       from pyvqnet.tensor import tensor
       from pyvqnet.tensor import QTensor
       shape = [2, 3]
       t = tensor.randu(shape)
       print(t)

randn
=====

.. py:function:: pyvqnet.tensor.randn(shape, mean=0.0,std=1.0, device=pyvqnet.DEV_CPU, dtype=None, requires_grad=False)

   创建一个具有正态分布随机值的 QTensor 。

   :param shape: 要创建的 QTensor 的形状。
   :param mean: 分布的均值,默认: 0。
   :param max: 分布的方差,默认: 1。
   :param device: 要使用的设备,默认 = pyvqnet.DEV_CPU ,使用CPU设备。
   :param dtype: 参数的数据类型,defaults:None,使用默认数据类型:kfloat32,代表32位浮点数。
   :param requires_grad: 是否计算梯度,默认为False。

   :return: 输出 QTensor 。

   Example::

       from pyvqnet.tensor import tensor
       from pyvqnet.tensor import QTensor
       shape = [2, 3]
       t = tensor.randn(shape)
       print(t)

binomial
========

.. py:function:: pyvqnet.tensor.binomial(total_countst, probs)

   创建一个由 :attr:`total_count` 和 :attr:`probs` 参数化的二项分布。

   :param total_counts: 伯努利试验的次数。
   :param probs: 事件概率。

   :return:
       二项分布的 QTensor。

   Example::

       import pyvqnet.tensor as tensor

       a = tensor.randu([3,4])
       b = 1000

       c = tensor.binomial(b,a)
       print(c)

multinomial
===========

.. py:function:: pyvqnet.tensor.multinomial(t, num_samples)

   返回一个张量,其中每行包含 num_samples 个索引采样,来自位于张量输入的相应行中的多项式概率分布。

   :param t: 输入概率分布,仅支持浮点数。
   :param num_samples: 采样样本。

   :return:
        输出采样索引

   Examples::

       from pyvqnet import tensor
       weights = tensor.QTensor([0.1,10, 3, 1])
       idx = tensor.multinomial(weights,3)
       print(idx)

       from pyvqnet import tensor
       weights = tensor.QTensor([0,10, 3, 2.2,0.0])
       idx = tensor.multinomial(weights,3)
       print(idx)

triu
====

.. py:function:: pyvqnet.tensor.triu(t, diagonal=0)

   返回输入 t 的上三角矩阵,其余部分被设为0。

   :param t: 输入 QTensor。
   :param diagonal: 偏移量(主对角线为 0, 正数为向上偏移, 负数为向下偏移), 默认:0。

   :return: 输出 QTensor。

   Examples::

       from pyvqnet.tensor import tensor
       a = tensor.arange(1.0, 2 * 6 * 5 + 1.0).reshape([2, 6, 5])
       u = tensor.triu(a, 1)
       print(u)

tril
====

.. py:function:: pyvqnet.tensor.tril(t, diagonal=0)

   返回输入 t 的下三角矩阵,其余部分被设为0。

   :param t: 输入 QTensor。
   :param diagonal: 偏移量(主对角线为 0, 正数为向上偏移, 负数为向下偏移), 默认:0。

   :return: 输出 QTensor。

   Examples::

       from pyvqnet.tensor import tensor
       a = tensor.arange(1.0, 2 * 6 * 5 + 1.0).reshape([12, 5])
       u = tensor.tril(a, 1)
       print(u)

**********
 数学函数
**********

floor
=====

.. py:function:: pyvqnet.tensor.floor(t)

   返回一个新的 QTensor,其中元素为输入 QTensor 的向下取整。

   :param t: 输入 QTensor 。

   :return: 输出 QTensor 。

   Example::

       from pyvqnet.tensor import tensor

       t = tensor.arange(-2.0, 2.0, 0.25)
       u = tensor.floor(t)
       print(u)

ceil
====

.. py:function:: pyvqnet.tensor.ceil(t)

   返回一个新的 QTensor,其中元素为输入 QTensor 的向上取整。

   :param t: 输入 QTensor 。
   :return: 输出 QTensor 。

   Example::

       from pyvqnet.tensor import tensor

       t = tensor.arange(-2.0, 2.0, 0.25)
       u = tensor.ceil(t)
       print(u)

 
round
=====

.. py:function:: pyvqnet.tensor.round(t)

   返回一个新的 QTensor,其中元素为输入 QTensor 的四舍五入到最接近的整数.

   :param t: 输入 QTensor 。
   :return: 输出 QTensor 。

   Example::

       from pyvqnet.tensor import tensor

       t = tensor.arange(-2.0, 2.0, 0.4)
       u = tensor.round(t)
       print(u)

sort
====

.. py:function:: pyvqnet.tensor.sort(t, axis: int, descending=False, stable=True)

   按指定轴对输入 QTensor 进行排序。

   :param t: 输入 QTensor 。
   :param axis: 排序使用的轴。
   :param descending: 如果是True,进行降序排序,否则使用升序排序。默认:False,为升序。
   :param stable: 是否使用稳定排序,默认:True,为稳定排序。
   :return: 输出 QTensor 。

   Example::

       from pyvqnet.tensor import tensor
       from pyvqnet.tensor import QTensor
       import numpy as np
       a = np.random.randint(10, size=24).reshape(3,8).astype(np.float32)
       A = QTensor(a)
       AA = tensor.sort(A,1,False)
       print(AA)

argsort
=======

.. py:function:: pyvqnet.tensor.argsort(t, axis: int, descending=False, stable=True)

   对输入变量沿给定轴进行排序,输出排序好的数据的相应索引。

   :param t: 输入 QTensor 。
   :param axis: 排序使用的轴。
   :param descending: 如果是True,进行降序排序,否则使用升序排序。默认:False,为升序。
   :param stable: 是否使用稳定排序,默认:True,为稳定排序。
   :return: 输出 QTensor 。

   Example::

       from pyvqnet.tensor import tensor
       from pyvqnet.tensor import QTensor
       import numpy as np
       a = np.random.randint(10, size=24).reshape(3,8).astype(np.float32)
       A = QTensor(a)
       bb = tensor.argsort(A,1,False)
       print(bb)

topK
====

.. py:function:: pyvqnet.tensor.topK(t, k, axis=-1, if_descent=True)

   返回给定输入张量沿给定维度的 k 个最大元素。

   如果 if_descent 为 False,则返回 k 个最小元素。

   :param t: 输入 QTensor 。
   :param k: 取排序后的 k 的个数。
   :param axis: 要排序的维度。默认 = -1,最后一个轴。
   :param if_descent: 排序使用升序还是降序,默认:True,降序。

   :return: 新的 QTensor 。

   Examples::

       from pyvqnet.tensor import tensor, QTensor
       x = QTensor([
           24., 13., 15., 4., 3., 8., 11., 3., 6., 15., 24., 13., 15., 3., 3., 8., 7.,
           3., 6., 11.
       ])
       x=x.reshape([2, 5, 1, 2])
       x.requires_grad = True
       y = tensor.topK(x, 3, 1)
       print(y)

argtopK
=======

.. py:function:: pyvqnet.tensor.argtopK(t, k, axis=-1, if_descent=True)

   返回给定输入张量沿给定维度的 k 个最大元素的索引。

   如果 if_descent 为 False,则返回 k 个最小元素的索引。

   :param t: 输入 QTensor 。
   :param k: 取排序后的 k 的个数。
   :param axis: 要排序的维度。默认 = -1,最后一个轴。
   :param if_descent: 排序使用升序还是降序,默认:True,降序。

   :return: 新的 QTensor 。

   Examples::

       from pyvqnet.tensor import tensor, QTensor
       x = QTensor([
           24., 13., 15., 4., 3., 8., 11., 3., 6., 15., 24., 13., 15., 3., 3., 8., 7.,
           3., 6., 11.
       ])
       x=x.reshape([2, 5, 1, 2])
       x.requires_grad = True
       y = tensor.argtopK(x, 3, 1)
       print(y)

add
===

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

sub
===

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

mul
===

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

divide
======

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

sums
====

.. py:function:: pyvqnet.tensor.sums(t: pyvqnet.tensor.QTensor, axis: Optional[int] = None, keepdims=False)

   对输入的 QTensor 按 axis 设定的轴计算元素和,如果 axis 是None,则返回所有元素和。

   :param t: 输入 QTensor 。
   :param axis: 用于求和的轴,默认为None。
   :param keepdims: 输出张量是否保留了减小的维度。默认为False。
   :return: 输出 QTensor 。

   Example::

       from pyvqnet.tensor import tensor
       from pyvqnet.tensor import QTensor
       t = QTensor(([1, 2, 3], [4, 5, 6]))
       x = tensor.sums(t)
       print(x)

cumsum
======

.. py:function:: pyvqnet.tensor.cumsum(t, axis=-1)

   返回维度轴中输入元素的累积总和。

   :param t: 输入 QTensor 。
   :param axis: 计算的轴,默认 -1,使用最后一个轴。
   :return: 输出 QTensor 。

   Example::

       from pyvqnet.tensor import tensor, QTensor
       t = QTensor(([1, 2, 3], [4, 5, 6]))
       x = tensor.cumsum(t,-1)
       print(x)

mean
====

.. py:function:: pyvqnet.tensor.mean(t: pyvqnet.tensor.QTensor, axis=None, keepdims=False)

   对输入的 QTensor 按 axis 设定的轴计算元素的平均,如果 axis 是None,则返回所有元素平均。

   :param t: 输入 QTensor ,需要是浮点数或者复数。
   :param axis: 用于求平均的轴,默认为None。
   :param keepdims: 输出张量是否保留了减小的维度。默认为False。
   :return: 输出 QTensor 或 均值。

   Example::

       from pyvqnet.tensor import tensor
       from pyvqnet.tensor import QTensor
       t = QTensor([[1, 2, 3], [4, 5, 6.0]])
       x = tensor.mean(t, axis=1)
       print(x)

median
======

.. py:function:: pyvqnet.tensor.median(t: pyvqnet.tensor.QTensor, axis=None, keepdims=False)

   对输入的 QTensor 按 axis 设定的轴计算元素的平均,如果 axis 是None,则返回所有元素平均。

   :param t: 输入 QTensor 。
   :param axis: 用于求平均的轴,默认为None。
   :param keepdims: 输出张量是否保留了减小的维度。默认为False。
   :return: 输出 QTensor 或 中值。

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

std
===

.. py:function:: pyvqnet.tensor.std(t: pyvqnet.tensor.QTensor, axis=None, keepdims=False, unbiased=True)

   对输入的 QTensor 按 axis 设定的轴计算元素的标准差,如果 axis 是None,则返回所有元素标准差。

   :param t: 输入 QTensor 。
   :param axis: 用于求标准差的轴,默认为None。
   :param keepdims: 输出张量是否保留了减小的维度。默认为False。
   :param unbiased: 是否使用贝塞尔修正,默认:True。
   :return: 输出 标准差。

   Example::

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

var
===

.. py:function:: pyvqnet.tensor.var(t: pyvqnet.tensor.QTensor, axis=None, keepdims=False, unbiased=True)

   对输入的 QTensor 按 axis 设定的轴计算元素的方差,如果 axis 是None,则返回所有元素方差。

   :param t: 输入 QTensor 。
   :param axis: 用于求方差的轴,默认为None。
   :param keepdims: 输出张量是否保留了减小的维度。默认为False。
   :param unbiased: 是否使用贝塞尔修正,默认:True。
   :return: 输出方差。

   Example::

       from pyvqnet.tensor import tensor
       from pyvqnet.tensor import QTensor

       a = QTensor([[-0.8166, -1.3802, -0.3560]])
       a_var = tensor.var(a)
       print(a_var)

matmul
======

.. py:function:: pyvqnet.tensor.matmul(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

   二维矩阵点乘或3、4维张量进行批矩阵乘法,或一维向量与二维矩阵矩阵向量积,或两个一维向量点积。

   :param t1: 第一个 QTensor 。
   :param t2: 第二个 QTensor 。
   :return:  输出 QTensor 。

   Example::

       from pyvqnet.tensor import tensor
       t1 = tensor.ones([2,3])
       t1.requires_grad = True
       t2 = tensor.ones([3,4])
       t2.requires_grad = True
       t3  = tensor.matmul(t1,t2)
       t3.backward(tensor.ones_like(t3))
       print(t1.grad)
       print(t2.grad)

kron
====

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

einsum
======

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

       vqneta = tensor.randn((20,30,40,50))
       z = tensor.einsum('...ij->...ji', vqneta)
       print(z.shape)

reciprocal
==========

.. py:function:: pyvqnet.tensor.reciprocal(t)

   计算输入 QTensor 的倒数。

   :param t: 输入 QTensor 。

   :return:  输出 QTensor 。

   Example::

       from pyvqnet.tensor import tensor

       t = tensor.arange(1, 10, 1)
       u = tensor.reciprocal(t)
       print(u)

 
sign
====

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

neg
===

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

trace
=====

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

exp
===

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

acos
====

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

asin
====

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

atan
====

.. py:function:: pyvqnet.tensor.atan(t: pyvqnet.tensor.QTensor)

   计算输入 t 每个元素的反正切。

   :param t: 输入 QTensor 。

   :return:  输出 QTensor 。

   Example::

       from pyvqnet.tensor import tensor

       t = tensor.arange(-1, 1, .5)
       u = tensor.atan(t)
       print(u)

sin
===

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

cos
===

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

tan
===

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

tanh
====

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

sinh
====

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

cosh
====

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

power
=====

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

abs
===

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

log
===

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

log_softmax
===========

.. py:function:: pyvqnet.tensor.log_softmax(t, axis=-1)

   顺序计算在轴axis上的softmax函数以及log函数的结果。

   :param t: 输入 QTensor 。
   :param axis: 用于求softmax的轴,默认为-1。

   :return: 输出 QTensor。

   Example::

       from pyvqnet import tensor
       output = tensor.arange(1,13).reshape([3,2,2])
       t = tensor.log_softmax(output,1)
       print(t)

sqrt
====

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

square
======

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

eigh
====

.. py:function:: pyvqnet.tensor.eigh(t: QTensor)

   返回复厄米矩阵(共轭对称)或实对称矩阵的特征值和特征向量。
   返回两个对象,一个包含a的特征值的一维数组,
   以及相应特征向量(以列表示)的二维方阵或矩阵(取决于输入类型)。

   :param: 输入QTensor。
   :param: t的特征值和特征向量。
   :return:

       返回特征值以及特征向量

   Examples::

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

frobenius_norm
==============

.. py:function:: pyvqnet.tensor.frobenius_norm(t: QTensor, axis: int = None, keepdims=False)

   对输入的 QTensor 按 axis 设定的轴计算张量的F范数,如果 axis 是None,则返回所有元素F范数。

   :param t: 输入 QTensor 。
   :param axis: 用于求F范数的轴,默认为None。
   :param keepdims: 输出张量是否保留了减小的维度。默认为False。
   :return: 输出 QTensor 或 F范数值。


   Example::

       from pyvqnet.tensor import tensor,QTensor
       t = QTensor([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]],
                   [[13., 14., 15.], [16., 17., 18.]]])
       t.requires_grad = True
       result = tensor.frobenius_norm(t, -2, False)
       print(result)

***********
 逻辑函数
***********

maximum
=======

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

minimum
=======

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

min
===

.. py:function:: pyvqnet.tensor.min(t: pyvqnet.tensor.QTensor, axis=None, keepdims=False)

   对输入的 QTensor 按 axis 设定的轴计算元素的最小值,如果 axis 是None,则返回所有元素的最小值。

   :param t: 输入 QTensor 。
   :param axis: 用于求最小值的轴,默认为None。
   :param keepdims: 输出张量是否保留了减小的维度。默认为False。

   :return: 输出 QTensor 或浮点数。

   Example::

       from pyvqnet.tensor import tensor
       from pyvqnet.tensor import QTensor
       t = QTensor([[1, 2, 3], [4, 5, 6]])
       x = tensor.min(t, axis=1, keepdims=True)
       print(x)

max
===

.. py:function:: pyvqnet.tensor.max(t: pyvqnet.tensor.QTensor, axis=None, keepdims=False)

   对输入的 QTensor 按 axis 设定的轴计算元素的最大值,如果 axis 是None,则返回所有元素的最大值。

   :param t: 输入 QTensor 。
   :param axis: 用于求最大值的轴,默认为None。
   :param keepdims: 输出张量是否保留了减小的维度。默认为False。

   :return: 输出 QTensor 或浮点数。


   Example::

       from pyvqnet.tensor import tensor
       from pyvqnet.tensor import QTensor
       t = QTensor([[1, 2, 3], [4, 5, 6]])
       x = tensor.max(t, axis=1, keepdims=True)
       print(x)

clip
====

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

where
=====

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

nonzero
=======

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

isfinite
========

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

isinf
=====

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

isnan
=====

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

isneginf
========

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

isposinf
========

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

logical_and
===========

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

logical_or
==========

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

logical_not
===========

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

logical_xor
===========

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

greater
=======

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

greater_equal
=============

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

less
====

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

less_equal
==========

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

equal
=====

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

not_equal
=========

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

bitwise_and
===========

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

************
 变换函数
************

broadcast
=========

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

select
======

.. py:function:: pyvqnet.tensor.select(t: pyvqnet.tensor.QTensor, index)

   输入字符串形式的索引位置,获取该索引下的数据切片,返回一个新的 QTensor 。

   :param t: 输入 QTensor 。
   :param index: 一个字符串包含切片的索引。
   :return: 输出 QTensor 。

   Example::

       from pyvqnet.tensor import tensor
       from pyvqnet.tensor import QTensor
       import numpy as np
       t = QTensor(np.arange(1,25).reshape(2,3,4))

       indx = [":", "0", ":"]
       t.requires_grad = True
       t.zero_grad()
       ts = tensor.select(t,indx)

       print(ts)

concatenate
===========

.. py:function:: pyvqnet.tensor.concatenate(args: list, axis=1)

   对 args 内的多个 QTensor 沿 axis 轴进行联结,返回一个新的 QTensor 。

   :param args: 包含输入 QTensor 。
   :param axis: 要连接的维度。 必须介于 0 和输入张量的最大维数之间，默认:1。
   :return: 输出 QTensor 。

   Example::

       from pyvqnet.tensor import tensor
       from pyvqnet.tensor import QTensor
       x = QTensor([[1, 2, 3],[4,5,6]], requires_grad=True)
       y = 1-x
       x = tensor.concatenate([x,y],1)
       print(x)

stack
=====

.. py:function:: pyvqnet.tensor.stack(QTensors: list, axis)

   沿新轴 axis 堆叠输入的 QTensors ,返回一个新的 QTensor。

   :param QTensors: 包含输入 QTensor 。
   :param axis: 要堆叠的维度。 必须介于 0 和输入张量的最大维数之间。
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

permute
=======

.. py:function:: pyvqnet.tensor.permute(t: pyvqnet.tensor.QTensor, dim: list)

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
       tt = tensor.permute(t,[2,0,1])
       print(tt)

transpose
=========

.. py:function:: pyvqnet.tensor.transpose(t: pyvqnet.tensor.QTensor, dim: list)

   根据输入的 dim 的顺序,改变t 的轴的顺序。如果 dims = None,则按顺序反转 t 的轴。该函数功能与 permute 一致。

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

tile
====

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

squeeze
=======

.. py:function:: pyvqnet.tensor.squeeze(t: pyvqnet.tensor.QTensor, axis: int = - 1)

   删除 axis 指定的轴,该轴的维度为1。如果 axis = None ,则将输入所有长度为1的维度删除。

   :param t: 输入 QTensor 。
   :param axis: 要压缩的轴,默认为None。
   :return: 输出 QTensor 。

   Example::

       from pyvqnet.tensor import tensor
       from pyvqnet.tensor import QTensor
       import numpy as np
       a = np.arange(6).reshape(1,6,1).astype(np.float32)
       A = QTensor(a)
       AA = tensor.squeeze(A,0)
       print(AA)

unsqueeze
=========

.. py:function:: pyvqnet.tensor.unsqueeze(t: pyvqnet.tensor.QTensor, axis: int = 0)

   在axis 指定的维度上插入一个维度为的1的轴,返回一个新的 QTensor 。

   :param t: 输入 QTensor 。
   :param axis: 要插入维度的位置,默认为0。
   :return: 输出 QTensor 。

   Example::

       from pyvqnet.tensor import tensor
       from pyvqnet.tensor import QTensor
       import numpy as np
       a = np.arange(24).reshape(2,1,1,4,3).astype(np.float32)
       A = QTensor(a)
       AA = tensor.unsqueeze(A,1)
       print(AA)

moveaxis
========

.. py:function:: pyvqnet.tensor.moveaxis(t, source: int, destination: int)

   将 `t` 的维度从 `source` 中的位置移动到 `destination` 中的位置。

   `t` 的其他未明确移动的维度保持其原始顺序,并出现在 `destination` 中未指定的位置。

   :param t: 输入 QTensor。
   :param source: (整数或整数元组)要移动的维度的原始位置。这些位置必须是唯一的。
   :param destination: (整数或整数元组)每个原始维度的目标位置。这些位置也必须是唯一的。

   :return: 新的QTensor

   Example::

       from pyvqnet import QTensor,tensor
       a = tensor.arange(0,24).reshape((2,3,4))
       b = tensor.moveaxis(a,(1, 2), (0, 1))
       print(b.shape)

swapaxis
========

.. py:function:: pyvqnet.tensor.swapaxis(t, axis1: int, axis2: int)

   交换输入 t 的 第 axis1 和 axis 维度。

   :param t: 输入 QTensor 。
   :param axis1: 要交换的第一个轴。
   :param axis2:  要交换的第二个轴。
   :return: 输出 QTensor 。

   Example::

       from pyvqnet.tensor import tensor
       from pyvqnet.tensor import QTensor

       import numpy as np
       a = np.arange(24).reshape(2,3,4).astype(np.float32)
       A = QTensor(a)
       AA = tensor.swapaxis(A, 2, 1)
       print(AA)

masked_fill
===========

.. py:function:: pyvqnet.tensor.masked_fill(t, mask, value)

   在 mask == 1 的位置,用值 value 填充输入。
   mask的形状必须与输入的 QTensor 的形状是可广播的。

   :param t: 输入 QTensor。
   :param mask: 掩码 QTensor,必须是kbool。
   :param value: 填充值。
   :return: 一个 QTensor。

   Examples::

       from pyvqnet.tensor import tensor
       import numpy as np
       a = tensor.ones([2, 2, 2, 2])
       mask = np.random.randint(0, 2, size=4).reshape([2, 2])
       b = tensor.QTensor(mask==1)
       c = tensor.masked_fill(a, b, 13)
       print(c)

flatten
=======

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

reshape
=======

.. py:function:: pyvqnet.tensor.reshape(t: pyvqnet.tensor.QTensor,new_shape)

   改变 QTensor 的形状,返回一个新的张量。

   :param t: 输入 QTensor 。
   :param new_shape: 新的形状。

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

flip
====

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

gather
======

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

scatter
=======

.. py:function:: pyvqnet.tensor.scatter(input, dim, index,src)

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

broadcast_to
============

.. py:function:: pyvqnet.tensor.broadcast_to(t, ref)

   受到某些约束,数组 t 被“广播”到参考形状,以便它们具有兼容的形状。

   https://numpy.org/doc/stable/user/basics.broadcasting.html

   :param t: 输入QTensor
   :param ref: 参考形状。

   :return: 广播后的 QTensor。

   Example::

       from pyvqnet.tensor.tensor import QTensor
       from pyvqnet.tensor import *
       ref = [2,3,4]
       a = ones([4])
       b = tensor.broadcast_to(a,ref)
       print(b.shape)

dense_to_csr
============

.. py:function:: pyvqnet.tensor.dense_to_csr(t)

   将稠密矩阵转化为CSR格式稀疏矩阵,仅支持2维。

   :param t: 输入稠密QTensor
   :return: CSR稀疏矩阵

   Example::

       from pyvqnet.tensor import QTensor,dense_to_csr

       a = QTensor([[2, 3, 4, 5]])
       b = dense_to_csr(a)
       print(b.csr_members())

csr_to_dense
============

.. py:function:: pyvqnet.tensor.csr_to_dense(t)

   将CSR格式稀疏矩阵转化为稠密矩阵,仅支持2维。

   :param t: 输入CSR稀疏矩阵
   :return: 稠密QTensor

   Example::

       from pyvqnet.tensor import QTensor,dense_to_csr,csr_to_dense

       a = QTensor([[2, 3, 4, 5]])
       b = dense_to_csr(a)
       c = csr_to_dense(b)
       print(c)


***********
 实用函数
***********

to_tensor
=========

.. py:function:: pyvqnet.tensor.to_tensor(x)

   将输入数值或 numpy.ndarray 等转换为 QTensor 。

   :param x: 整数、浮点数、布尔数、复数、或 numpy.ndarray
   :return: 输出 QTensor

   Example::

       from pyvqnet.tensor import tensor

       t = tensor.to_tensor(10.0)
       print(t)


pad_sequence
============

.. py:function:: pyvqnet.tensor.pad_sequence(qtensor_list, batch_first=False, padding_value=0)

   用 ``padding_value`` 填充可变长度张量列表。 ``pad_sequence`` 沿新维度堆叠张量列表,并将它们填充到相等的长度。
   输入是列表大小为 ``L x *`` 的序列。 L 是可变长度。

   :param qtensor_list: `list[QTensor]`- 可变长度序列列表。
   :param batch_first: 'bool' - 如果为真,输出将是 ``批大小 x 最长序列长度 x *`` ,否则为 ``最长序列长度 x 批大小 x *`` 。 默认值: False。
   :param padding_value: 'float' - 填充值。 默认值:0。

   :return:
       如果 batch_first 为 ``False``,则张量大小为 ``批大小 x 最长序列长度 x *``。
       否则张量的大小为 ``最长序列长度 x 批大小 x *`` 。

   Examples::

       from pyvqnet.tensor import tensor
       a = tensor.ones([4, 2,3])
       b = tensor.ones([1, 2,3])
       c = tensor.ones([2, 2,3])
       a.requires_grad = True
       b.requires_grad = True
       c.requires_grad = True
       y = tensor.pad_sequence([a, b, c], True)

       print(y)

pad_packed_sequence
===================

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

   Examples::

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

       print(lens_unpacked)

pack_pad_sequence
=================

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

   Examples::

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


       print(data.batch_sizes)

no_grad
=======

.. py:function:: pyvqnet.no_grad()

   禁用前向计算时记录反向传播节点。

   Example::

       import pyvqnet.tensor as tensor
       from pyvqnet import no_grad

       with no_grad():
           x = tensor.QTensor([1.0, 2.0, 3.0],requires_grad=True)
           y = tensor.tan(x)
           y.backward()
       #RuntimeError: output requires_grad is False.
