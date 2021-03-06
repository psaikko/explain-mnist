#!/usr/bin/env python3
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

rows, cols = 28, 28

x_train_flat = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test_flat = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

model = Sequential()

model.add(Dense(10, input_shape=(28*28,), activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(10, activation='softmax'))

y_train_cat = keras.utils.to_categorical(y_train, num_classes=10)
y_test_cat = keras.utils.to_categorical(y_test, num_classes=10)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train_flat, y_train_cat, epochs=10, batch_size=32)
score = model.evaluate(x_test_flat, y_test_cat, batch_size=32)

print("Scores:",score)

wts = model.get_weights()
np.save("hidden_weights_1", wts[0])
np.save("hidden_bias_1", wts[1])
np.save("hidden_weights_2", wts[2])
np.save("hidden_bias_2", wts[3])
np.save("output_weights", wts[4])
np.save("output_bias", wts[5])

y_pred = model.predict(x_test_flat, batch_size=32)
np.save("X", x_test_flat)
np.save("Y", y_test_cat)
np.save("Y_pred", y_pred)

print("Sample outputs:")
for p in y_pred[:3]:
    print(list(map(lambda x: round(x,2), p)))