import numpy as np

timesteps = 100  # 输入序列的时间步数
input_features = 32  # 输入特征空间的维度
output_features = 64  # 输出特征空间的维度

# 输入数据：随机噪声，仅作为示例
inputs = np.random.random((timesteps, input_features))

# 初始状态：全零向量
state_t = np.zeros((output_features,))

# 创建随机的权重矩阵
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))

successive_outputs = []
# input_t 是形状为(input_features,) 的向量
for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    successive_outputs.append(output_t)
    state_t = output_t
# 最终输出是一个形状为(timesteps,output_features) 的二维张量
final_output_sequence = np.stack(successive_outputs, axis=0)


