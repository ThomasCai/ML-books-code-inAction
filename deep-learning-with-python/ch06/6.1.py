import numpy as np
import string
from keras.preprocessing.text import Tokenizer
from keras.datasets import imdb
from keras import preprocessing  # this is a fault in the book.
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
import os
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

need_train = False

# 6.1.1 单词和字符的 one-hot 编码
# 单词级的 one-hot 编码(简单示例)
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
token_index = {}
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1
max_length = 10
results = np.zeros(shape=(len(samples), max_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1.
print("单词级的 one-hot 编码(简单示例): ", results)

# 字符级的 one-hot 编码(简单示例)
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
characters = string.printable  # 所有可打印的 ASCII 字符
token_index = dict(zip(range(1, len(characters) + 1), characters))
max_length = 50
results = np.zeros((len(samples), max_length, max(token_index.keys()) + 1))
for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        index = token_index.get(character)
        results[i, j, index] = 1.
print("字符级的 one-hot 编码(简单示例): ", results)

# 用 Keras 实现单词级的 one-hot 编码
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
tokenizer = Tokenizer(num_words=1000)  # 创建一个分词器(tokenizer),设置为只考虑前 1000 个最常见的单词
tokenizer.fit_on_texts(samples)  # 构建单词索引
sequences = tokenizer.texts_to_sequences(samples)  # 将字符串转换为整数索引组成的列表
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')  # 也可以直接得到 one-hot 二进制表示。
# 这个分词器也支持除 one-hot 编码外的其他向量化模式
word_index = tokenizer.word_index  # 找回单词索引
print('Found %s unique tokens.' % len(word_index))  # is 9 because keras can combine upper and lower.

# 使用散列技巧的单词级的 one-hot 编码(简单示例)
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
dimensionality = 1000
max_length = 10
results = np.zeros((len(samples), max_length, dimensionality))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word)) % dimensionality
        results[i, j, index] = 1.
print("使用散列技巧的单词级的 one-hot 编码(简单示例): ", results)

# 6.1.2 使用词嵌入
# 加载 IMDB 数据,准备用于 Embedding 层
max_features = 10000  # 作为特征的单词个数
maxlen = 20  # 在这么多单词后截断文本(这些单词都属于前 max_features 个最常见的单词)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# 将整数列表转换成形状为 (samples,maxlen) 的二维整数张量
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

if need_train:
    # 1. 利用 Embedding 层学习词嵌入
    # 在 IMDB 数据上使用 Embedding 层和分类器
    model = Sequential()
    model.add(Embedding(10000, 8, input_length=maxlen))
    model.add(Flatten())  # 将三维的嵌入张量展平成形状为 (samples, maxlen * 8) 的二维张量
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    model.summary()

    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=32,
                        validation_split=0.2)
    result = history.history
    print('acc: ', np.mean(result['acc']))
    print('val_acc: ', np.mean(result['val_acc']))

# 6.1.3 整合在一起:从原始文本到词嵌入
# 1. 下载 IMDB 数据的原始文本
imdb_dir = '/home/thomas/Downloads/keras_ch06_data/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')
labels = []
texts = []
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

# 2. 对数据进行分词
# 对 IMDB 原始数据的文本进行分词
maxlen = 100  # 在 100 个单词后截断评论
training_samples = 200  # 在 200 个样本上训练
validation_samples = 10000  # 在 10 000 个样本上验证
max_words = 10000  # 只考虑数据集中前 10 000 个最常见的单词

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index  # this is the whole word index.
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)  # 将数据划分为训练集和验证集,但首先要打乱数据,因为一开始数据中的
# 样本是排好序的(所有负面评论都在前面,然后是所有正面评论)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

# 4. 对嵌入进行预处理
# 解析 GloVe 词嵌入文件
glove_dir = '/home/thomas/Downloads/glove.6B'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

# 准备 GloVe 词嵌入矩阵
embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# 模型定义
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# 将预训练的词嵌入加载到 Embedding 层中
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')

# 绘制结果
print(history.history.keys())
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#  在不使用预训练词嵌入的情况下,训练相同的模型
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))

# 绘制结果
print(history.history.keys())
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# 对测试集数据进行分词
test_dir = os.path.join(imdb_dir, 'test')
labels = []
texts = []
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(test_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)
sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)

# 在测试集上评估模型
model.load_weights('pre_trained_glove_model.h5')
result = model.evaluate(x_test, y_test)
print(result)
