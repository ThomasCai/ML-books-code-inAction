from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt


# 设置第一次运行或第二次运行
the_first_time = True
the_second_time = False

# 3.4.1 加载数据
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print(train_data[0])
print(train_labels[0])

# 由于限定为前10 000 个最常见的单词，单词索引都不会超过10 000
print(max([max(sequence) for sequence in train_data]))

# 解码评论
word_index = imdb.get_word_index() # word_index 是一个将单词映射为整数索引的字典
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) # 键值颠倒，将整数索引映射为单词
# 将评论解码。注意，索引减去了3，因为0、1、2是为“padding”（填充）、“start of sequence”（序列开始）、“unknown”（未知词）分别保留的索引
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

# 3.4.2 准备数据
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension)) # 创建一个形状为(len(sequences), dimension) 的零矩阵
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1. # 将results[i] 的指定索引设为1
    return results
x_train = vectorize_sequences(train_data) # 将训练数据向量化
x_test = vectorize_sequences(test_data) # 将测试数据向量化

print("训练数据向量化后的结果：", x_train[0])

y_train = np.asarray(train_labels).astype('float32') # 训练标签向量化
y_test = np.asarray(test_labels).astype('float32') # 测试标签向量化

# 分离训练、验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

"""
配置优化器 及 传入自定义的损失函数或指标函数

from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=0.001), 
loss='binary_crossentropy', 
metrics=['accuracy'])

from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
loss=losses.binary_crossentropy,
metrics=[metrics.binary_accuracy])
"""
if the_first_time:
    # 3.4.3 构建网络
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])  # 编译模型
    # 3.4.4　验证你的方法
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=20,
                        batch_size=512,
                        validation_data=(x_val, y_val))
    # history 它是一个字典，包含训练过程中的所有数据。
    history_dict = history.history
    print(history_dict.keys())

    # 绘制训练损失和验证损失
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 绘制训练精度和验证精度
    plt.clf()
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

"""
在这种情况下，为了防止过拟合，你可以在3 轮之后停止训练。通常来说，你可以使用许
多方法来降低过拟合，我们将在第4 章中详细介绍。
我们从头开始训练一个新的网络，训练4 轮，然后在测试数据上评估模型。
"""
if the_second_time:
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=4, batch_size=512)
    # 验证结果
    results = model.evaluate(x_test, y_test)
    print(results)
    # 预测结果
    print(model.predict(x_test))

"""
3.4.6　进一步的实验
通过以下实验，你可以确信前面选择的网络架构是非常合理的，虽然仍有改进的空间。
 前面使用了两个隐藏层。你可以尝试使用一个或三个隐藏层，然后观察对验证精度和测
试精度的影响。
 尝试使用更多或更少的隐藏单元，比如 32 个、64 个等。
 尝试使用 mse损失函数代替 binary_crossentropy。
 尝试使用 tanh激活（这种激活在神经网络早期非常流行）代替relu。
"""






