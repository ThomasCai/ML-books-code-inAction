import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt


# 标量（0D张量）
x1 = np.array(12)
print(x1)
print(x1.ndim)

# 向量（1D张量）
x2 = np.array([12, 3, 6, 14, 7])
print(x2)
print(x2.ndim) # 1D张量，5D向量。维度既可以表示张量，也可以表示向量

# 矩阵（2D张量）
x3 = np.array([[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]])
print(x3)
print(x3.ndim)

# 3D张量与更高维张量
x4 = np.array([[[5, 78, 2, 34, 0],
                [6, 79, 3, 35, 1],
                [7, 80, 4, 36, 2]],
               [[5, 78, 2, 34, 0],
                [6, 79, 3, 35, 1],
                [7, 80, 4, 36, 2]],
               [[5, 78, 2, 34, 0],
                [6, 79, 3, 35, 1],
                [7, 80, 4, 36, 2]]])
print(x4)
print(x4.ndim)

# 显示图片
(train_images, train_labels), (test_images, test_labels) = mnist.load_data() # 如果失败，参照readme第二章
print(train_images.ndim)
print(train_images.shape)
print(train_images.dtype)

digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
