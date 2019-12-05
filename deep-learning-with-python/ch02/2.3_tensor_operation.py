import keras
import numpy as np


# 等价于 output = relu(dot(W, input) + b) W为2D张量 b为向量 dot为点积
keras.layers.Dense(512, activation='relu')

# 2.3.1 逐元素运算
def naive_relu(x):
    assert len(x.shape) == 2
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x

def naive_add(x, y):
    assert len(x.shape) == 2
    assert x.shape == y.shape
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x

z = x + y  # 等价于naive_add 逐元素相加
z = np.maximum(z, 0.)  # 等价于naive_relu 逐元素的relu

# 2.3.2 广播
def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
    return x

x = np.random.random((64, 3, 32, 10)) # x 是形状为(64, 3, 32, 10) 的随机张量
y = np.random.random((32, 10)) # y 是形状为(32, 10) 的随机张量
z = np.maximum(x, y) # 输出z 的形状是(64, 3, 32, 10)，与x 相同

# 2.3.3 张量点积
z = np.dot(x, y) # z = x·y

# 点积的实现细节-两个向量的实现
def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z 

# 点积的实现细节-矩阵和向量的实现
def naive_matrix_vector_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
    return z

# 点积的实现细节-复用两个向量实现
def naive_matrix_vector_dot(x, y):
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        z[i] = naive_vector_dot(x[i, :], y)
    return z

# 点积的实现细节-两个矩阵实现
def naive_matrix_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]
    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] = naive_vector_dot(row_x, column_y)
    return z

# 2.3.4 张量变形
x = np.array([[0., 1.],
              [2., 3.],
              [4., 5.]])
print(x.shape)
x = x.reshape((6,1)) # 重置形状：注意有两层括号
print(x)
x = x.reshape((2,3))
print(x)

x = np.zeros((300, 20))
x = np.transpose(x) # 转置
print(x.shape)
