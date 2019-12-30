import keras

# ModelCheckpoint 与EarlyStopping 回调函数
callbacks_list = [
keras.callbacks.EarlyStopping(
monitor='acc',
patience=1,
),
keras.callbacks.ModelCheckpoint(
filepath='my_model.h5',
monitor='val_loss',
save_best_only=True,
)
]
model.compile(optimizer='rmsprop',
loss='binary_crossentropy',
metrics=['acc'])
model.fit(x, y,
epochs=10,
batch_size=32,
callbacks=callbacks_list,
validation_data=(x_val, y_val))

# ReduceLROnPlateau 回调函数
callbacks_list = [
keras.callbacks.ReduceLROnPlateau(
monitor='val_loss'
factor=0.1,
patience=10,
)
]
model.fit(x, y,
epochs=10,
batch_size=32,
callbacks=callbacks_list,
validation_data=(x_val, y_val))