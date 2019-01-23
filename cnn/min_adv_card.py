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

convo_weights = np.load("convo_weights_1.npy")
convo_bias = np.load("convo_bias_1.npy")
output_weights = np.load("output_weights.npy")
output_bias = np.load("output_bias.npy")

#mip_solver.objective.set_sense(mip_solver.objective.sense.minimize)

input_dim  = 28
n_kernels  = 16
pool_dim   = 4
kernel_dim = 3

kernel_res_w = 28 - kernel_dim + 1
kernel_res_h = 28 - kernel_dim + 1

output_nodes = 10

# output variables
mip_solver.variables.add(
    lb = [-cplex.infinity]*output_nodes,
    ub = [cplex.infinity]*output_nodes,
    types = "C"*output_nodes,
    names = ["o%d"%i for i in range(output_nodes)])

# input variables
mip_solver.variables.add(
    lb    = [0]*input_dim*input_dim, 
    ub    = [1]*input_dim*input_dim,
    types = "C"*input_dim*input_dim,
    names = ["x(%d,%d)" % (i,j) for i in range(input_dim) for j in range(input_dim)])

# for each kernel
for i in range(n_kernels):
    # output variable for (n-d)*(n-d) pixels
    for yi in range(kernel_res_h):
        for xi in range(kernel_res_w):
            # add convolution output variable
            mip_solver.variables.add(
                lb = [-cplex.infinity],
                ub = [cplex.infinity],
                types = "C",
                names = ["co(%d,%d,%d)" % (yi,xi,i)]
            )

# for each kernel
for i in range(n_kernels):
    # for each avgpool
    for yi in range(kernel_res_h//pool_dim):
        for xi in range(kernel_res_w//pool_dim):
            # add avgpool output variable
            mip_solver.variables.add(
                lb = [0],
                ub = [cplex.infinity],
                types = "C",
                names = ["y(%d,%d,%d)" % (yi,xi,i)]
            )

            # add relu variables
            mip_solver.variables.add(
                lb    = [0], 
                ub    = [cplex.infinity],
                types = "C",
                names = ["s(%d,%d,%d)"  % (yi,xi,i)])
            mip_solver.variables.add(
                lb    = [0], 
                ub    = [1],
                types = "B",
                names = ["z(%d,%d,%d)" % (yi,xi,i)])
    
            # add relu indicator constraints
            mip_solver.indicator_constraints.add(
                indvar="z(%d,%d,%d)" % (yi,xi,i),
                complemented=1,
                rhs=0.0,
                sense="E",
                lin_expr=(["y(%d,%d,%d)" % (yi,xi,i)], [1.0]),
                name="ind(%d,%d,%d)1" % (yi,xi,i),
                indtype=mip_solver.indicator_constraints.type_.if_)

            mip_solver.indicator_constraints.add(
                indvar="z(%d,%d,%d)" % (yi,xi,i),
                complemented=0,
                rhs=0.0,
                sense="E",
                lin_expr=(["s(%d,%d,%d)" % (yi,xi,i)], [1.0]),
                name="ind(%d,%d,%d)0" % (yi,xi,i),
                indtype=mip_solver.indicator_constraints.type_.if_)

# for each kernel
for i in range(n_kernels):
    # for each output
    for yi in range(kernel_res_h):
        for xi in range(kernel_res_w):
            cplex_vars = []
            cplex_coefs = []

            for yii in range(kernel_dim):
                for xii in range(kernel_dim):
                    cplex_coefs.append(float(convo_weights[yii,xii,0,i]))
                    cplex_vars.append("x(%d,%d)" % (yi+yii,xi+xii))
            
            cplex_vars.append("co(%d,%d,%d)" % (yi,xi,i))
            cplex_coefs.append(-1)

            mip_solver.linear_constraints.add(
                lin_expr = [[cplex_vars, cplex_coefs]],
                senses   = "E",
                rhs      = [float(-convo_bias[i])],
                names    = ["convo_sum_(%d,%d,%d)" % (yi,xi,i)]
            )

# for each kernel
for i in range(n_kernels):
    # for each pool
    for yi in range(kernel_res_h//pool_dim):
        for xi in range(kernel_res_w//pool_dim):
            cplex_vars = []
            cplex_coefs = []

            for yii in range((yi)*pool_dim,(yi+1)*pool_dim):
                for xii in range((xi)*pool_dim,(xi+1)*pool_dim):
                    cplex_vars.append("co(%d,%d,%d)" % (yii,xii,i))
                    cplex_coefs.append(1.0/(pool_dim*pool_dim))

            cplex_vars += ["y(%d,%d,%d)" % (yi,xi,i), "s(%d,%d,%d)" % (yi,xi,i)]
            cplex_coefs += [-1, 1]

            mip_solver.linear_constraints.add(
                lin_expr = [[cplex_vars, cplex_coefs]],
                senses   = "E",
                rhs      = [0],
                names    = ["pool_relu_(%d,%d,%d)" % (yi,xi,i)]
            )        

flattened_vars = []

for yi in range(kernel_res_h//pool_dim):
    for xi in range(kernel_res_w//pool_dim):
        for i in range(n_kernels):
            flattened_vars.append("y(%d,%d,%d)" % (yi,xi,i))

# encode output layer 
for i in range(output_nodes):
    out_vars  = flattened_vars + ["o%d"%i]
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

# for test_index in range(10):
#     input_image = X[test_index]
#     prediction = max((cl,i) for (i,cl) in enumerate(Y_pred[test_index]))

#     #print(prediction)

#     real = max((cl,i) for (i,cl) in enumerate(Y[test_index]))
#     #print(real)

#     # plt.imshow(np.array(input_image).reshape((28,28)), cmap="gray")
#     # plt.show()

#     # fix variables in mus and cube
#     mip_solver.variables.set_lower_bounds([
#         ("x%d"%i, x)  for (i,x) in enumerate(input_image)
#     ])

#     mip_solver.variables.set_upper_bounds([
#         ("x%d"%i, x)  for (i,x) in enumerate(input_image)
#     ])

#     try:
#         mip_solver.write("debug.lp")
#         mip_solver.solve()
#         output = mip_solver.solution.get_values(["o%d"%i for i in range(output_nodes)])
#         mip_prediction = max((cl,i) for (i,cl) in enumerate(output))[1]
#         #print(output)
#         print(mip_prediction)
#     except CplexError as e:
#         print(e)

test_index = 0
input_image = X[test_index]
prediction = max((cl,i) for (i,cl) in enumerate(Y_pred[test_index]))

mip_solver.variables.add(
    obj   = [1]*input_dim*input_dim,
    lb    = [0]*input_dim*input_dim, 
    ub    = [1]*input_dim*input_dim,
    types = "B"*input_dim*input_dim,
    names = ["xz(%d,%d)" % (i,j) for i in range(input_dim) for j in range(input_dim)])

mip_solver.objective.set_sense(mip_solver.objective.sense.maximize)

for j in range(input_dim):
    for i in range(input_dim):
        mip_solver.indicator_constraints.add(
            indvar="xz(%d,%d)" % (i,j),
            complemented=0,
            rhs=float(input_image[i,j]),
            sense="E",
            lin_expr=(["x(%d,%d)" % (i,j)], [1]),
            name="xind(%d,%d)-1" % (i,j),
            indtype=mip_solver.indicator_constraints.type_.if_)

mip_solver.write("debug.lp")

for target in range(10):

    ct = 0
    for cl in range(10):
        if target != cl:
            mip_solver.linear_constraints.add(
                lin_expr = [[["o%d"%target, "o%d"%cl], [1,-1]]],
                senses   = "G",
                rhs      = [0],
                names    = ["obj_constr_%d" % ct]
            )
            ct += 1

    try:
        mip_solver.solve()
        print(mip_solver.solution.get_status_string())
        print("prediction",mip_solver.solution.get_values(["o%d"%i for i in range(10)]))

    except CplexError as e:
        print(e)

    mip_solver.linear_constraints.delete(["obj_constr_%d"%i for i in range(ct)])