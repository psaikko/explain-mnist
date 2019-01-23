#!/usr/bin/env python3
import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

X = np.load("X.npy")
Y = np.load("Y.npy")

hidden_weights_1 = np.load("hidden_weights_1.npy")
hidden_bias_1 = np.load("hidden_bias_1.npy")
hidden_weights_2 = np.load("hidden_weights_2.npy")
hidden_bias_2 = np.load("hidden_bias_2.npy")
output_weights = np.load("output_weights.npy")
output_bias = np.load("output_bias.npy")

print(X.shape)
print(hidden_weights_1.shape)
print(hidden_bias_1.shape)
print(hidden_weights_2.shape)
print(hidden_bias_2.shape)
print(output_weights.shape)
print(output_bias.shape)

def from_onehot(x):
  return max((cl,i) for (i,cl) in enumerate(x))[1]

def predict(x):
    hidden_out_1 = x @ hidden_weights_1 + hidden_bias_1
    hidden_out_1 = hidden_out_1 * (hidden_out_1 > 0)    # relu

    hidden_out_2 = hidden_out_1 @ hidden_weights_2 + hidden_bias_2
    hidden_out_2 = hidden_out_2 * (hidden_out_2 > 0)    # relu

    output = hidden_out_2 @ output_weights + output_bias
    print(output)
    return from_onehot(output)

X = np.load("output.npy")
print([predict(x) for x in X[:10]])
print(list(map(from_onehot, Y[:10])))

