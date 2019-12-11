# explain
Experiments in explainable AI with exact optimization tools on the MNIST image dataset. 

Dependencies: 
- python3
- CPLEX (and docplex python library)
- matplotlib
- numpy
- keras 

## [twoclass](./twoclass)
Proof-of-concept level scripts on a simple neural network and a binary classification task.

`train_twoclass.py`

Train a simple fully connected neural network with one hidden layer.

`min_explanation.py`

Compute an "explanation" of a prediciton. Given an input image this is a minimal set of pixels which determine the output label, regardless the value of any other pixels. 
Uses CPLEX as a decision procedure for a "destructive MUS" algorithm.

![](./img/explanation.png)

`min_adv_sum.py`

Compute the smallest adversarial example with respect to sum of squared errors
on the original picture, using mixed integer quadratic programming (MIQP).

![](./img/min_adversarial_sse.png)

`min_adv_card.py` 

Compute a smallest adversarial example with respect the total number of changed 
pixels from the original input using mixed integer programming (MIP).

![](./img/min_adversarial_card.png)


## [multiclass](./multiclass)

Apply the above techniques to a multiclass classifier.

`train_multiclass.py`

Train a somewhat more complicated neural network 
with multiple hidden layers and output classes.

`min_adv_sum.py`

For a given imput image, compute the minimal changes 
to predict each possible label.

![](./img/adversarial_multiclass.png)

## [cnn](./cnn)

Can we do the same with simple convolutional networks?

`train_cnn_simple.py`

Train a simple CNN with 10 3x3 convolution kernels.

`min_adv_sum.py`

As above, but we observe clear visual differences in the minimum adversarial changes compared to the network without convolution.

![](./img/adversarial_convolution_0.png)
![](./img/adversarial_convolution_1.png)

`min_adv_card.py`

Computing minimum number of changed pixels instead.

![](./img/adv_conv_sq_0.png)
![](./img/adv_conv_sq_1.png)

## [binary](./binary)

`binary.py`

Implement some encodings of AAAI paper [Verifying Properties of Binarized Deep Neural Networks](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16898/16241)
- MIP: working
- IP:  working
- CNF: producing intractably large formulas