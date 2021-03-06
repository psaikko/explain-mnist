#!/usr/bin/env python3
import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

X = np.load("X.npy")
Y = np.load("Y.npy")

# Test that we can load weights from file
hidden_weights = np.load("hidden_weights.npy")
hidden_bias = np.load("hidden_bias.npy")
output_weights = np.load("output_weights.npy")
output_bias = np.load("output_bias.npy")

# Test our interpretation of the NN structure
def predict(x):
    hidden_out = x @ hidden_weights + hidden_bias
    hidden_out = hidden_out * (hidden_out > 0)    # relu
    output = hidden_out @ output_weights + output_bias
    return (output[0] > 0)*1

correct = 0
for x,y in zip(X,Y):
  if y == predict(x): correct += 1
print("Accuracy: %.4f" % (correct / len(Y)))
