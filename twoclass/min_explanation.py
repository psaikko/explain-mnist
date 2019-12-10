#!/usr/bin/env python3

import cplex
from cplex import Cplex
from cplex.exceptions import CplexError
import numpy as np
import matplotlib.pyplot as plt

mip_solver = Cplex()

mip_solver.set_results_stream(None)
mip_solver.set_warning_stream(None)
mip_solver.set_error_stream(None)
mip_solver.parameters.threads.set(1)

hidden_weights = np.load("hidden_weights.npy")
hidden_bias = np.load("hidden_bias.npy")
output_weights = np.load("output_weights.npy")
output_bias = np.load("output_bias.npy")

mip_solver.objective.set_sense(mip_solver.objective.sense.minimize)

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

test_index = 0

# for (input_image, output_label) in zip(X,Y):

mus = set()
cube = set(range(input_dim))
while len(cube) > 0:
    test_feature = list(cube)[0]
    cube.remove(test_feature)

    input_image = X[test_index]
    prediction  = 1 if Y_pred[test_index] > 0.5 else 0
    #print(input_image)

    #print("NN prediction",prediction)
    #break

    # fix variables in mus and cube
    mip_solver.variables.set_lower_bounds([
      ("x%d"%i, x)  for (i,x) in enumerate(input_image) if i in mus.union(cube)
    ])

    mip_solver.variables.set_upper_bounds([
      ("x%d"%i, x)  for (i,x) in enumerate(input_image) if i in mus.union(cube)
    ])

    # unfix all others
    mip_solver.variables.set_lower_bounds([
      ("x%d"%i, 0) for i in range(input_dim) if i not in mus.union(cube)
    ])

    mip_solver.variables.set_upper_bounds([
      ("x%d"%i, 1) for i in range(input_dim) if i not in mus.union(cube)
    ])
    
    # can we make prediction of opposite class?

    #print("testing prediction", 1 - prediction)

    if prediction == 1:
        mip_solver.variables.set_upper_bounds([("output", 0)])
        mip_solver.variables.set_lower_bounds([("output", -cplex.infinity)])
    else:
        mip_solver.variables.set_lower_bounds([("output", 0)])
        mip_solver.variables.set_upper_bounds([("output", cplex.infinity)])

    #mip_solver.write("debug.lp")

    try:
        mip_solver.solve()
        if mip_solver.solution.get_status_string() == "integer infeasible":
            raise CplexError()
        # ok
        #mus.add(test_feature)
        #output = mip_solver.solution.get_values(["output"])
        #print("solution exists with output", output)
        mus.add(test_feature)
        print("MUS", test_feature)
    except CplexError:
        # no solution
        print("DROP", test_feature)

print(len(mus))

plt.imshow(X[test_index].reshape((28,28)), cmap='gray')
expl = np.array([1 if i in mus else 0 for (i,x) in enumerate(X[0])])
plt.imshow(expl.reshape((28,28)), cmap='summer', alpha=0.6)
plt.show()