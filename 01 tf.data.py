# '''
# Author: CHENJIE
# Date: 2020-09-16 15:21:00
# LastEditTime: 2020-09-18 08:32:16
# LastEditors: Please set LastEditors
# Description: 1.实现了tf.data的创建数据集Dataset的两种方法
#              2.还有 
# FilePath: /tf_codeLearning/01 tf.data.py
# '''

import tensorflow as tf
import numpy as np
print('tensorflow-version:{}'.format(tf.__version__))
print(tf.__version__)
print("-----------------------------------")
#1.这里使用的是一维的列表,创建dataset
dataset = tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6])#这种列表和下面的np.array的方式是完全一样的
dataset_array = tf.data.Dataset.from_tensor_slices(np.array([1,2,3,4,5,6]))
# rom_tensor_slice 会将其中的元素都转化为为tensor数据类型
print("转化后的数据类型为：")
print(dataset)
print("-----------------------------------")
print("将TensorSliceDataset中的每一个元素进行输出：")
for ele in dataset:
    print(ele)
print("-----------------------------------")
# print("将TensorSliceDataset中的每一个tensor转成numpy数值")
for ele in dataset:
    print(ele.numpy())

print("-----------------------------------")
#2.这里使用的是二维的列表创建dataset
dataset = tf.data.Dataset.from_tensor_slices([[1,2],[3,4],[5,6]])
print(dataset)#可以看到其shape为2，<TensorSliceDataset shapes: (2,), types: tf.int32>
for ele in dataset:
    print(ele)
# 结果为
# < TensorSliceDataset shapes: (2,), types: tf.int32 >
# tf.Tensor([1 2], shape=(2,), dtype=int32)
# tf.Tensor([3 4], shape=(2,), dtype=int32)
# tf.Tensor([5 6], shape=(2,), dtype=int32)
for ele in dataset:
    print(ele.numpy())
print("-----------------------------------")

#使用字典的形式创建dataset
print("使用字典的形式创建dataset")
dataset_dict = tf.data.Dataset.from_tensor_slices({'a' :[1,2,3,4],
                                                   'b':[6,7,8,9],
                                                   'c':[10,11,12,13]})
print(dataset_dict)
for ele in dataset_dict:
    print(ele)

print("-----------------------------------")
# 使用data.take方法来获取其中额定数目的值
# dataset为[1,2,3,4,5,6])
print("使用data.take方法来获取其中额定数目的值，原本的dataset为[1,2,3,4,5,6])")
dataset = tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6])#
for ele in dataset.take(4):
    print(ele.numpy())
print("-----------------------------------")
# 将数据集中的数据进行乱序排列，因为如果每一个epoch都是相同的数据顺序，模型不会关注数据之间的关系，而是数据本身，所以乱序是必须要做的事情
#使用shuffle方法来做，dataset.shuffle
print('使用shuffle来进行对数据集乱序')
dataset = dataset.shuffle(6)
for ele in dataset:
    print(ele.numpy())
print("-----------------------------------")
print('使用shuffle和repeat来进行对数据集多次乱序')#乱序还会配合着repeat这个参数使用，即重复的打乱
dataset = dataset.shuffle(6)
dataset = dataset.repeat(count=3)#里面的参数为空就是无限循环
for ele in dataset:
    print(ele.numpy())
#在现实情况中，由于数据集都非常大，内存没办法一次性的读取进来，所以将其分成一个一个的batch
print("-----------------------------------")
print('使用batch来分开数据集')
dataset = tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6])#
dataset = dataset.batch(3)#即一个batch的长度为3，值与内存和数据集大小相关
for ele in dataset:
    print(ele.numpy())
# result为
# [1 2 3]
# [4 5 6]
print("-----------------------------------")
print("使用一个函数map对Dataset进行平方")
dataset = tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6])#
dataset = dataset.map(tf.square)
for ele in dataset:
    print(ele.numpy())
