# 重要函数
# 1.Conv2D
Conv2D用于实现卷积，需要设定通道数、卷积核大小、padding方式、激活函数、名称等。
使用方法如下：
```python
# 建立模型
x = Conv2D(32, kernel_size= 5,padding = 'same',activation="relu")(inputs)
```
# 2、MaxPooling2D
Conv2D用于实现最大池化，需要设定池化核大小、padding方式、名称等。
```python
x = MaxPooling2D(pool_size = 2, strides = 2, padding = 'same',)(x)
```
#全部代码
```python
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
```
# 输出如下
```python
Epoch 1/5
60000/60000 [==============================] - 64s 1ms/sample - loss: 0.1061 - accuracy: 0.9677
Epoch 2/5
60000/60000 [==============================] - 65s 1ms/sample - loss: 0.0523 - accuracy: 0.9841
Epoch 3/5
60000/60000 [==============================] - 65s 1ms/sample - loss: 0.0415 - accuracy: 0.9876
Epoch 4/5
60000/60000 [==============================] - 65s 1ms/sample - loss: 0.0348 - accuracy: 0.9899
Epoch 5/5
60000/60000 [==============================] - 65s 1ms/sample - loss: 0.0287 - accuracy: 0.9916
10000/10000 - 2s - loss: 0.0441 - accuracy: 0.9887
```
