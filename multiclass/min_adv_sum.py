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
# mip_solver.parameters.threads.set(1)

hidden_weights = [np.load("hidden_weights_1.npy"), np.load("hidden_weights_2.npy")]
hidden_bias = [np.load("hidden_bias_1.npy"), np.load("hidden_bias_2.npy")]
output_weights = np.load("output_weights.npy")
output_bias = np.load("output_bias.npy")

#mip_solver.objective.set_sense(mip_solver.objective.sense.minimize)

input_dim = 28*28
hidden_nodes = [10,10]
output_nodes = 10

# output variables
mip_solver.variables.add(
    lb = [-cplex.infinity]*output_nodes,
    ub = [cplex.infinity]*output_nodes,
    types = "C"*output_nodes,
    names = ["o%d"%i for i in range(output_nodes)])

# input variables
mip_solver.variables.add(
    lb    = [0]*input_dim, 
    ub    = [1]*input_dim,
    types = "C"*input_dim,
    names = ["x%d" % i for i in range(input_dim)])

# hidden layer variables
for i,n in enumerate(hidden_nodes):
    mip_solver.variables.add(
        lb    = [0]*n, 
        ub    = [cplex.infinity]*n,
        types = "C"*n,
        names = ["y(%d,%d)" % (i,j) for j in range(n)])
    mip_solver.variables.add(
        lb    = [0]*n, 
        ub    = [cplex.infinity]*n,
        types = "C"*n,
        names = ["s(%d,%d)" % (i,j) for j in range(n)])
    mip_solver.variables.add(
        lb    = [0]*n, 
        ub    = [1]*n,
        types = "B"*n,
        names = ["z(%d,%d)" % (i,j) for j in range(n)])

# relu indicator constraints
for i in range(len(hidden_nodes)):
    for j in range(hidden_nodes[i]):
        mip_solver.indicator_constraints.add(
            indvar="z(%d,%d)" % (i,j),
            complemented=1,
            rhs=0.0,
            sense="E",
            lin_expr=(["y(%d,%d)" % (i,j)], [1.0]),
            name="ind(%d,%d)1" % (i,j),
            indtype=mip_solver.indicator_constraints.type_.if_)

        mip_solver.indicator_constraints.add(
            indvar="z(%d,%d)" % (i,j),
            complemented=0,
            rhs=0.0,
            sense="E",
            lin_expr=(["s(%d,%d)" % (i,j)], [1.0]),
            name="ind(%d,%d)0" % (i,j),
            indtype=mip_solver.indicator_constraints.type_.if_)

# encode hidden layers
for i in range(hidden_nodes[0]):
    a_i = hidden_weights[0][:,i]
    cplex_vars  = ["x%d"%j for j in range(input_dim)] + ["y(0,%d)" % i, "s(0,%d)" % i]
    cplex_coefs = list(a_i) + [-1, 1]
    cplex_coefs = [float(v) for v in cplex_coefs]

    mip_solver.linear_constraints.add(
        lin_expr = [[cplex_vars, cplex_coefs]],
        senses   = "E",
        rhs      = [float(-hidden_bias[0][i])],
        names    = ["hidden_sum_(0,%d)" % i]
    )

for i in range(hidden_nodes[1]):
    a_i = hidden_weights[1][:,i]
    cplex_vars  = ["y(0,%d)"%j for j in range(hidden_nodes[0])] + ["y(1,%d)" % i, "s(1,%d)" % i]
    cplex_coefs = list(a_i) + [-1, 1]
    cplex_coefs = [float(v) for v in cplex_coefs]

    mip_solver.linear_constraints.add(
        lin_expr = [[cplex_vars, cplex_coefs]],
        senses   = "E",
        rhs      = [float(-hidden_bias[1][i])],
        names    = ["hidden_sum_(1,%d)" % i]
    )

# encode output layer 
for i in range(output_nodes):
    out_vars  = ["y(1,%d)"%j for j in range(hidden_nodes[1])] + ["o%d"%i]
    out_coefs = list(output_weights[:,i]) + [-1]
    out_coefs = [float(v) for v in out_coefs]

    mip_solver.linear_constraints.add(
        lin_expr = [[out_vars, out_coefs]],
        senses   = "E",
        rhs      = [float(-output_bias[i])],
        names    = ["output_sum_%d"%i]
    )

X = np.load("X.npy")
Y = np.load("Y.npy")
Y_pred = np.load("Y_pred.npy")

test_index = 0
input_image = X[test_index]
prediction = max((cl,i) for (i,cl) in enumerate(Y_pred[test_index]))

mip_solver.variables.add(
    lb    = [0]*input_dim, 
    ub    = [1]*input_dim,
    types = "C"*input_dim,
    names = ["xi%d" % i for i in range(input_dim)])

mip_solver.variables.add(
    #obj   = [1]*input_dim,
    lb    = [0]*input_dim, 
    ub    = [1]*input_dim,
    types = "C"*input_dim,
    names = ["xd%d" % i for i in range(input_dim)])

# fix variables in mus and cube
mip_solver.variables.set_lower_bounds([
    ("xi%d"%i, x)  for (i,x) in enumerate(input_image)
])

mip_solver.variables.set_upper_bounds([
    ("xi%d"%i, x)  for (i,x) in enumerate(input_image)
])

# x_input + x_diff = x
for i in range(input_dim):
    mip_solver.linear_constraints.add(
        lin_expr = [[["xi%d"%i,"xd%d"%i,"x%d"%i], [1,1,-1.0]]],
        senses   = "E",
        rhs      = [0],
        names    = ["diff %d" % i]
    )

# minimize sum_i x_diff_i^2
mip_solver.parameters.optimalitytarget.set(
    mip_solver.parameters.optimalitytarget.values.optimal_global)
mip_solver.objective.set_sense(mip_solver.objective.sense.minimize)
mip_solver.objective.set_quadratic_coefficients([
    ["xd%d"%i, "xd%d"%i, 2] for i in range(input_dim)
])

images = []

# Loop over target labels for input_image
for target in range(10):
    ct = 0
    for cl in range(10):
        # For classes other than target, constrain output variable to small value
        if target != cl:
            mip_solver.linear_constraints.add(
                lin_expr = [[["o%d"%target, "o%d"%cl], [1,-1]]],
                senses   = "G",
                rhs      = [1e-6],
                names    = ["obj_constr_%d" % ct]
            )
            ct += 1
    try:
        mip_solver.solve()
        print(mip_solver.solution.get_status_string())
        opt = mip_solver.solution.get_objective_value()
        print(opt)
        print("Change",input_dim - opt,"pixels")

        vs = mip_solver.solution.get_values(["x%d"%i for i in range(input_dim)])
        images.append(np.array(vs))
        
    except CplexError as e:
        print(e)

    # remove constraint on output
    mip_solver.linear_constraints.delete(["obj_constr_%d"%i for i in range(ct)])
    
np.save("output", np.array(images))

plt.subplots(2,10)
for i,img in enumerate(images):
    plt.subplot(2,10,i+1)
    plt.imshow(img.reshape((28,28)), cmap='gray')
    plt.title("Predict %d"%(i))
for i,img in enumerate(images):
    plt.subplot(2,10,10+i+1)
    diff = img - input_image
    plt.imshow(diff.reshape((28,28)), cmap='summer')
    plt.title("Diff")
plt.show()
