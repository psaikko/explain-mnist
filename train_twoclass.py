#!/usr/bin/env python3
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

classes = [1,3]

x_train, y_train = map(np.array, zip(*[(x,y) for (x,y) in zip(x_train, y_train) if y in classes]))
x_test, y_test = map(np.array, zip(*[(x,y) for (x,y) in zip(x_test, y_test) if y in classes]))

x_train = x_train / 255
x_test = x_test / 255

y_train = np.floor_divide(y_train, 3)
y_test = np.floor_divide(y_test, 3)

print(y_train[:10])

x_train_flat = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test_flat = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

model = Sequential()
model.add(Dense(20, input_dim=784, activation="relu"))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train_flat, y_train, epochs=10, batch_size=32)

score = model.evaluate(x_test_flat, y_test, batch_size=32)

print(score)

#https://stackoverflow.com/questions/41908379/keras-plot-training-validation-and-test-set-accuracy

# plt.subplots(1,2)
# #  "Accuracy"
# plt.subplot(1,2,1)
# plt.plot(history.history['acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# # "Loss"
# plt.subplot(1,2,2)
# plt.plot(history.history['loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.show()

y_pred = model.predict(x_test_flat, batch_size=32)

wts = model.get_weights()

np.save("hidden_weights", wts[0])
np.save("hidden_bias", wts[1])
np.save("output_weights", wts[2])
np.save("output_bias", wts[3])

np.save("X", x_test_flat)
np.save("Y", y_test)
np.save("Y_pred", y_pred)