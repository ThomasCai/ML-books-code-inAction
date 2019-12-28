from keras import layers
from keras import Input
from keras.models import Model
from keras import applications


# Inception 模块
branch_a = layers.Conv2D(128, 1,activation='relu', strides=2)(x)
branch_b = layers.Conv2D(128, 1, activation='relu')(x)
branch_b = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_b)
branch_c = layers.AveragePooling2D(3, strides=2)(x)
branch_c = layers.Conv2D(128, 3, activation='relu')(branch_c)
branch_d = layers.Conv2D(128, 1, activation='relu')(x)
branch_d = layers.Conv2D(128, 3, activation='relu')(branch_d)
branch_d = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_d)
output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)

# 残差连接
x = ...
y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
y = layers.add([y, x])

x = ...
y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
y = layers.MaxPooling2D(2, strides=2)(y)
residual = layers.Conv2D(128, 1, strides=2, padding='same')(x)
y = layers.add([y, residual])

# 共享层权重
lstm = layers.LSTM(32) # 将一个LSTM 层实例化一次
left_input = Input(shape=(None, 128))
left_output = lstm(left_input)
right_input = Input(shape=(None, 128))
right_output = lstm(right_input)
merged = layers.concatenate([left_output, right_output], axis=-1)
predictions = layers.Dense(1, activation='sigmoid')(merged)
model = Model([left_input, right_input], predictions)
model.fit([left_data, right_data], targets)

# 将模型作为层
y = model(x)

y1, y2 = model([x1, x2])

# 重复调用 如：双摄像头
# 图像处理基础模型是Xception 网络（只包括卷积基）
xception_base = applications.Xception(weights=None,include_top=False)
left_input = Input(shape=(250, 250, 3)) # 输入是250×250 的RGB 图像
right_input = Input(shape=(250, 250, 3))
left_features = xception_base(left_input) # 对相同的视觉模型调用两次
right_input = xception_base(right_input)
# 合并后的特征包含来自左右两个视觉输入中的信息
merged_features = layers.concatenate([left_features, right_input], axis=-1)
