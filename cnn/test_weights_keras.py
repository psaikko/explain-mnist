#!/usr/bin/env python3
import keras
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten, AvgPool2D
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

rows, cols = 28, 28
x_train = x_train.reshape(x_train.shape[0], rows, cols, 1)
x_test = x_test.reshape(x_test.shape[0], rows, cols, 1)
input_shape = (rows, cols, 1)

model = Sequential()
model.add(Conv2D(16, (3,3), input_shape=input_shape, activation="linear"))
model.add(AvgPool2D((4,4)))
model.add(Activation("relu"))
model.add(Flatten())
model.add(Dense(10, activation="softmax"))

y_train_cat = keras.utils.to_categorical(y_train, num_classes=10)
y_test_cat = keras.utils.to_categorical(y_test, num_classes=10)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

model.set_weights([
    np.load("convo_weights_1.npy"),
    np.load("convo_bias_1.npy"),
    np.load("output_weights.npy"),
    np.load("output_bias.npy"),
])

#score = model.evaluate(x_test, y_test_cat, batch_size=32)

#print(score)

#y_pred = model.predict(x_test, batch_size=32)

x = np.expand_dims(x_test[0], 0)
print(x.shape)

get_layer_output = K.function([model.layers[0].input], [model.layers[0].output])
layer_output = get_layer_output([x])[0]

print(layer_output)
print(layer_output.shape)

print(layer_output[0,:,:,0])


plt.subplots(4,4)
plt.title("keras")
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(layer_output[0,:,:,i])
plt.show()