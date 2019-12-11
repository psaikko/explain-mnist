#!/usr/bin/env python3

import cplex
from cplex import Cplex
from cplex.exceptions import CplexError
import numpy as np
import matplotlib.pyplot as plt

mip_solver = Cplex()

# mip_solver.set_results_stream(None)
# mip_solver.set_warning_stream(None)
# mip_solver.set_error_stream(None)
#mip_solver.parameters.threads.set(1)

hidden_weights = np.load("hidden_weights.npy")
hidden_bias = np.load("hidden_bias.npy")
output_weights = np.load("output_weights.npy")
output_bias = np.load("output_bias.npy")

#mip_solver.objective.set_sense(mip_solver.objective.sense.minimize)

input_dim = 28*28
hidden_nodes = 20

mip_solver.variables.add(
    lb = [-cplex.infinity],
    ub = [cplex.infinity],
    types = "C",
    names = ["output"])

mip_solver.variables.add(
    lb    = [0]*input_dim, 
    ub    = [1]*input_dim,
    types = "C"*input_dim,
    names = ["x%d" % i for i in range(input_dim)])

mip_solver.variables.add(
    lb    = [0]*hidden_nodes, 
    ub    = [cplex.infinity]*hidden_nodes,
    types = "C"*hidden_nodes,
    names = ["y%d" % i for i in range(hidden_nodes)])
mip_solver.variables.add(
    lb    = [0]*hidden_nodes, 
    ub    = [cplex.infinity]*hidden_nodes,
    types = "C"*hidden_nodes,
    names = ["s%d" % i for i in range(hidden_nodes)])
mip_solver.variables.add(
    lb    = [0]*hidden_nodes, 
    ub    = [1]*hidden_nodes,
    types = "B"*hidden_nodes,
    names = ["z%d" % i for i in range(hidden_nodes)])

for i in range(hidden_nodes):
    mip_solver.indicator_constraints.add(
        indvar="z%d" % i,
        complemented=1,
        rhs=0.0,
        sense="E",
        lin_expr=(["y%d" % i], [1.0]),
        name="ind%d-1" % i,
        indtype=mip_solver.indicator_constraints.type_.if_)

    mip_solver.indicator_constraints.add(
        indvar="z%d" % i,
        complemented=0,
        rhs=0.0,
        sense="E",
        lin_expr=(["s%d" % i], [1.0]),
        name="ind%d-0" % i,
        indtype=mip_solver.indicator_constraints.type_.if_)

# encode hidden layer
for i in range(hidden_nodes):
    a_i = hidden_weights[:,i]
    cplex_vars  = ["x%d"%j for j in range(input_dim)] + ["y%d" % i, "s%d" % i]
    cplex_coefs = list(a_i) + [-1, 1]
    cplex_coefs = [float(v) for v in cplex_coefs] # why?!

    mip_solver.linear_constraints.add(
        lin_expr = [[cplex_vars, cplex_coefs]],
        senses   = "E",
        rhs      = [float(-hidden_bias[i])],
        names    = ["hidden_sum %d" % i]
    )

# encode output layer 
out_vars  = ["y%d"%j for j in range(hidden_nodes)] + ["output"]
out_coefs = list(output_weights) + [-1]
out_coefs = [float(v) for v in out_coefs]

mip_solver.linear_constraints.add(
    lin_expr = [[out_vars, out_coefs]],
    senses   = "E",
    rhs      = [float(-output_bias[0])],
    names    = ["output_sum"]
)

X = np.load("X.npy")
Y = np.load("Y.npy")
Y_pred = np.load("Y_pred.npy")

test_index = 2

mip_solver.variables.add(
    obj   = [1]*input_dim,
    lb    = [0]*input_dim, 
    ub    = [1]*input_dim,
    types = "B"*input_dim,
    names = ["xz%d" % i for i in range(input_dim)])

input_image = X[test_index]

mip_solver.objective.set_sense(mip_solver.objective.sense.maximize)

for i in range(input_dim):
    mip_solver.indicator_constraints.add(
        indvar="xz%d" % i,
        complemented=0,
        rhs=float(input_image[i]),
        sense="E",
        lin_expr=(["x%d" % i], [1]),
        name="xind%d-1" % i,
        indtype=mip_solver.indicator_constraints.type_.if_)

if Y_pred[test_index] > 0.5:
    mip_solver.variables.set_upper_bounds([("output", -.1)])
    mip_solver.variables.set_lower_bounds([("output", -cplex.infinity)])
else:
    mip_solver.variables.set_lower_bounds([("output", .1)])
    mip_solver.variables.set_upper_bounds([("output", cplex.infinity)])

try:
    mip_solver.solve()
    print(mip_solver.solution.get_status_string())
    opt = mip_solver.solution.get_objective_value()
    print(opt)
    print("Changed",input_dim - opt,"pixels")
    print("Objective fn value:",mip_solver.solution.get_values(["output"]))

    plt.subplots(1,3)

    plt.subplot(1,3,1)
    plt.imshow(input_image.reshape((28,28)), cmap='gray')
    plt.title("Input image")

    plt.subplot(1,3,2)
    vs = mip_solver.solution.get_values(["xz%d"%i for i in range(input_dim)])
    plt.imshow(np.array(vs).reshape((28,28)), cmap='summer')
    plt.title("Unlocked pixels")

    plt.subplot(1,3,3)
    vs = mip_solver.solution.get_values(["x%d"%i for i in range(input_dim)])
    plt.imshow(np.array(vs).reshape((28,28)), cmap='gray')
    plt.title("Result image")

    plt.savefig("min_adversarial_card.png")
    plt.show()
    
except CplexError as e:
    print(e)
