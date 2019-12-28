from keras.models import Model
from keras import layers
from keras import Input
import numpy as np

text_vocabulary_size = 10000  # 一个文本片段（比如新闻文章）
question_vocabulary_size = 10000 # 一个自然语言描述的问题
answer_vocabulary_size = 500 # 用于回答问题的信息

# 构建多输入模型
text_input = Input(shape=(None,), dtype='int32', name='text') # 文本输入
embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input)
encoded_text = layers.LSTM(32)(embedded_text)

question_input = Input(shape=(None,),dtype='int32',name='question') # 问题
embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

concatenated = layers.concatenate([encoded_text, encoded_question],axis=-1) # 结合多个输入
answer = layers.Dense(answer_vocabulary_size,activation='softmax')(concatenated) # 连接分类器
model = Model([text_input, question_input], answer)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['acc'])

# 将数据输入到多输入模型中
num_samples = 1000
max_length = 100

# 随机生成数据
text = np.random.randint(1, text_vocabulary_size,size=(num_samples, max_length))
question = np.random.randint(1, question_vocabulary_size,size=(num_samples, max_length))
answers = np.random.randint(answer_vocabulary_size, size=(num_samples))

# 回答是one-hot 编码的，不是整数
answers = keras.utils.to_categorical(answers, answer_vocabulary_size)

# 使用输入组成的列表来拟合
model.fit([text, question], answers, epochs=10, batch_size=128)

# 使用输入组成的字典来拟合（只有对输入进行命名之后才能用这种方法）
# model.fit({'text': text, 'question': question}, answers, epochs=10, batch_size=128)
