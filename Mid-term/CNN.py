import tensorflow as tf
from tensorflow.keras.layers import Flatten,Conv2D,Dropout,Input,Dense,MaxPooling2D
from tensorflow.keras.models import Model
import numpy as np

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# 载入Mnist手写数据集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = np.expand_dims(x_train,-1)
x_test = np.expand_dims(x_test,-1)
# 作为输入
inputs = Input([28,28,1])
x = Conv2D(32, kernel_size= 5,padding = 'same',activation="relu")(inputs)
x = MaxPooling2D(pool_size = 2, strides = 2, padding = 'same',)(x)
x = Conv2D(64, kernel_size= 5,padding = 'same',activation="relu")(x)
x = MaxPooling2D(pool_size = 2, strides = 2, padding = 'same',)(x)
x = Flatten()(x)
x = Dense(1024)(x)
x = Dense(256)(x)
out = Dense(10, activation='softmax')(x)

# 建立模型
model = Model(inputs,out)

# 设定优化器，loss，计算准确率
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 利用fit进行训练
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)
