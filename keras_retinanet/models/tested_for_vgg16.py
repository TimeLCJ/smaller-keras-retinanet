from keras.callbacks import TensorBoard
import keras
import numpy as np
import vgg16
import time
from keras.models import Model

X = np.ones((4, 224, 224, 3))
y = np.ones((4, 1000))
inputs = keras.layers.Input(shape=(None, None, 3))
model = vgg16.VGG16(input_tensor=inputs, include_top=False)
model_name = "kaggle_cat_dog-cnn-64x2-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# model.fit(X, y, batch_size =4, epochs=10, callbacks=[tensorboard])

"""
查看某层的输出
"""
layer_model = Model(inputs=model.input,
                    outputs=model.get_layer('block5_pool').output)

#以这个model的预测值作为输出
output = layer_model.predict(X)
print(output.shape)

