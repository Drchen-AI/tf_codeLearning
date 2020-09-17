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
print("将TensorSliceDataset中的每一个tensor转成numpy数值")
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

