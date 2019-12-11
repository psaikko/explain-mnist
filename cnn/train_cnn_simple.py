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
model.add(Conv2D(10, (3,3), input_shape=input_shape, activation="linear"))
model.add(AvgPool2D((6,6)))
model.add(Activation("relu"))
model.add(Flatten())
model.add(Dense(10, activation="softmax"))

y_train_cat = keras.utils.to_categorical(y_train, num_classes=10)
y_test_cat = keras.utils.to_categorical(y_test, num_classes=10)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

history = model.fit(x_train, y_train_cat, epochs=10, batch_size=32)

score = model.evaluate(x_test, y_test_cat, batch_size=32)

print("Scores",score)

wts = model.get_weights()
np.save("convo_weights_1", wts[0])
np.save("convo_bias_1", wts[1])
np.save("output_weights", wts[2])
np.save("output_bias", wts[3])

y_pred = model.predict(x_test, batch_size=32)
np.save("X", x_test)
np.save("Y", y_test_cat)
np.save("Y_pred", y_pred)

