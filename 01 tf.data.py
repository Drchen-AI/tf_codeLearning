import tensorflow as tf
print('tensorflow-version:{}'.format(tf.__version__))
print(tf.__version__)
print("-----------------------------------")
#1.这里使用的是一维的列表
dataset = tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6])
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
#2.这里使用的是二维的列表
dataset = tf.data.Dataset.from_tensor_slices([[1,2],[3,4],[5,6]])
print(dataset)#可以看到其shape为2，<TensorSliceDataset shapes: (2,), types: tf.int32>
for ele in dataset:
    print(ele)
# 结果为
# < TensorSliceDataset shapes: (2,), types: tf.int32 >
# tf.Tensor([1 2], shape=(2,), dtype=int32)
# tf.Tensor([3 4], shape=(2,), dtype=int32)
# tf.Tensor([5 6], shape=(2,), dtype=int32)