### 无输入输出数据  并不能运行

from keras import layers
from keras import Input
from keras.models import Model

vocabulary_size = 50000
num_income_groups = 10

posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)
x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation='relu')(x)

# 连接分类器
age_prediction = layers.Dense(1, name='age')(x)
income_prediction = layers.Dense(num_income_groups,activation='softmax',name='income')(x)
gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)

# Model 类将输入张量和输出张量转换为一个模型
model = Model(posts_input,[age_prediction, income_prediction, gender_prediction])

# 编译模型 且 加入权重 以及 训练模型 以下两种选一种即可
model.compile(optimizer='rmsprop',loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'],loss_weights=[0.25, 1., 10.])
model.fit(posts, [age_targets, income_targets, gender_targets],epochs=10, batch_size=64)

model.compile(optimizer='rmsprop',loss={'age': 'mse','income': 'categorical_crossentropy','gender': 'binary_crossentropy'},loss_weights={'age': 0.25,'income': 1.,'gender': 10.})
model.fit(posts, {'age': age_targets,'income': income_targets,'gender': gender_targets},epochs=10, batch_size=64)