#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt

X = np.load("X.npy")
Y = np.load("Y.npy")
Y_pred = np.load("Y_pred.npy")

convo_weights  = np.load("convo_weights_1.npy")
convo_bias     = np.load("convo_bias_1.npy")
output_weights = np.load("output_weights.npy")
output_bias    = np.load("output_bias.npy")

print(convo_weights.shape)   # 3, 3, 1, 16
print(convo_bias.shape)      # 16,
print(output_weights.shape)  # 576, 10
print(output_bias.shape)     # 10,

n_kernels  = 16
pool_dim   = 4
kernel_dim = 3

kernel_res_w = 28 - kernel_dim + 1
kernel_res_h = 28 - kernel_dim + 1

def predict(x):

  kernel_out = np.zeros(( kernel_res_h, kernel_res_w, n_kernels ))
  for k in range(n_kernels):
    for yi in range(kernel_res_h):
      for xi in range(kernel_res_w):
        input_area = x[ yi:(yi+kernel_dim), xi:(xi+kernel_dim), 0 ]
        kernel = convo_weights[:,:,0,k]
        t = input_area * kernel
        s = np.sum(t) + convo_bias[k]
        kernel_out[yi][xi][k] = s
  print(kernel_out.shape) # 26, 26, 16

  # plt.subplots(4,4)
  # plt.title("test")
  # for i in range(16):
  #     plt.subplot(4,4,i+1)
  #     plt.imshow(kernel_out[:,:,i])
  # plt.show()
  # exit()

  avg_pool_out = np.zeros(( kernel_res_h//pool_dim, kernel_res_w//pool_dim, n_kernels ))
  for k in range(n_kernels):
    for yi in range(kernel_res_h//pool_dim):
      for xi in range(kernel_res_w//pool_dim):
        input_area = kernel_out[ (yi*pool_dim):((yi+1)*pool_dim), (xi*pool_dim):((xi+1)*pool_dim), k ]
        s = np.sum(input_area) / (pool_dim*pool_dim)
        avg_pool_out[yi][xi][k] = s

  print(avg_pool_out.shape) # 6, 6, 16
  flat = avg_pool_out.flatten()
  flat = flat * (flat > 0) # relu

  output = flat @ output_weights + output_bias
  return np.argmax(output)

predict(X[0])

print([predict(x) for x in X[:10]])
print([np.argmax(y) for y in Y[:10]])
print([np.argmax(y) for y in Y_pred[:10]])
