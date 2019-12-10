from keras.datasets import reuters
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt


# 设置第一次运行或第二次运行
the_first_time = False
the_second_time = False
the_third_time = True

# 3.5.1 加载路透社数据
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
print(len(train_data))
print(len(train_labels))
# 与IMDB 评论一样，每个样本都是一个整数列表（表示单词索引）。
print(train_data[10])
# 将索引解码为新闻文本
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print('训练数据集第0条评论文本：\n', decoded_newswire)
# 话题索引编号
print(train_labels[10])

# 3.5.2 准备数据
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# 数据向量化（one-hot编码）
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

# 标签向量化（one-hot编码）
one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)
"""
注意，Keras 内置方法可以实现这个操作，你在MNIST 例子中已经见过这种方法。
from keras.utils.np_utils import to_categorical
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)
"""

# 留出验证集
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

if the_first_time:
    # 3.5.3 构建网络
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', 
                  metrics=['accuracy']) # 编译模型
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
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(partial_x_train,
              partial_y_train,
              epochs=9,
              batch_size=512,
              validation_data=(x_val, y_val))
    results = model.evaluate(x_test, one_hot_test_labels)
    print(results)
    # 预测结果
    print(model.predict(x_test))
    # 在新数据上生成预测结果
    predictions = model.predict(x_test)
    print(predictions[0].shape)
    print(np.sum(predictions[0]))
    print(np.argmax(predictions[0]))

# 完全随机的分类器
import copy
test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels) == np.array(test_labels_copy)
print(float(np.sum(hits_array)) / len(test_labels))

# 验证中间层维度足够大的重要性
if the_third_time:
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(partial_x_train,
              partial_y_train,
              epochs=20,
              batch_size=128,
              validation_data=(x_val, y_val))
    results = model.evaluate(x_test, one_hot_test_labels)
    print(results)


