#!/usr/bin/env python3
import numpy as np
import docplex.mp.model as model

from keras.datasets import mnist

from pysat.solvers import Glucose4
from pysat.formula import CNF, CNFPlus
from pysat.card import CardEnc
from pysat.card import EncType
import pysat

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

class Block:
    def __init__(self, in_dim, out_dim, weights, layer):
        self.layer = layer
        self.A = weights[0]
        self.b = weights[1]
        self.alpha = weights[2]
        self.mu = weights[3]
        self.sigma = np.sqrt(weights[4])
        self.gamma = weights[5]
    
        self.in_dim = in_dim
        self.out_dim = out_dim

        assert self.A.shape == (out_dim, in_dim)
        assert self.alpha.shape == (out_dim,)
        assert self.mu.shape == (out_dim,)
        assert self.sigma.shape == (out_dim,)
        assert self.gamma.shape == (out_dim,)

    def evaluate(self, x):

        assert x.shape == (self.in_dim,)

        y = self.A @ x + self.b
        assert y.shape == (self.out_dim,)

        z = self.alpha * ((y - self.mu)/self.sigma) + self.gamma
        return np.sign(z)

    def to_cnf(self, x_vars, cnf):
        
        y_vars = [cnf.nv + i + 1 for i in range(self.out_dim)]
        cnf.nv = max(y_vars)

        for i in range(self.out_dim):
            C_i = -(self.sigma[i] / self.alpha[i]) * self.gamma[i] + self.mu[i] - self.b[i]
            if self.alpha[i] > 0:
                C_i = np.ceil(C_i)
            elif self.alpha[i] < 0:
                C_i = np.floor(C_i)

            print(self.sigma[i], self.alpha[i], self.gamma[i], self.mu[i], self.b[i])
            print(C_i)
            print(np.sum(self.A[i]))

            C_ = np.ceil(C_i / 2 + np.sum(self.A[i]) / 2)
            
            n_nega = sum(self.A[i] > 0)

            print(C_, n_nega)

            D = int(C_ + n_nega)

            print(D)

            lits = [x if a > 0 else -x for (x,a) in zip(x_vars, self.A[i])]
        
            amo = CardEnc.atmost(lits, D-1, cnf.nv, EncType.seqcounter)
            
            r = amo.nv # TODO: Check assumption that this is "r(n, D)" of AAAI paper

            print("Adding", len(amo.clauses), "AMO clauses")

            cnf.extend(amo.clauses)
            cnf.append([-r,  y_vars[i]])
            cnf.append([ r, -y_vars[i]])

            # ale = CardEnc.atleast(lits, D, cnf.nv, EncType.seqcounter)

            # r = ale.nv # TODO: Check assumption that this is "r(n, D)" of AAAI paper

            # print("Adding", len(ale.clauses), "ALE clauses")

            print(len(cnf.clauses))

            # cnf.extend(ale.clauses)
            # cnf.append([-r,  y_vars[i]])
            # cnf.append([ r, -y_vars[i]])
        return y_vars

    def to_ip(self, x_vars, m):
        i_vars = [m.binary_var(name="i(%d,%d)"%(self.layer, i)) for i in range(self.out_dim)]
        v_vars = [m.integer_var(-1, 1, "v(%d,%d)"%(self.layer, i)) for i in range(self.out_dim)]

        for i in range(self.out_dim):
            C_i = -(self.sigma[i] / self.alpha[i]) * self.gamma[i] + self.mu[i] - self.b[i]
            if self.alpha[i] > 0:
                C_i = np.ceil(C_i)
            elif self.alpha[i] < 0:
                C_i = np.floor(C_i)
                
            if self.alpha[i] == 0: # TODO: float comparison errors?
                s = np.sign(self.gamma[i])
                v_vars[i].set_lb(s)
                v_vars[i].set_ub(s)
            else:
                ax = np.dot(x_vars, self.A[i])
                m.add_indicator(i_vars[i], ax >= C_i, 1)
                m.add_indicator(i_vars[i], ax <= C_i, 0)

                m.add_if_then(i_vars[i] == 1, v_vars[i] == 1)
                m.add_if_then(i_vars[i] == 0, v_vars[i] == -1)

        return v_vars
    
    def to_mip(self, x_vars, m):
        y_vars = [m.continuous_var(lb=-m.infinity, ub=m.infinity, name="y(%d,%d)"%(self.layer, i)) for i in range(self.out_dim)]

        for i in range(self.out_dim):
            m.add_constraint(np.dot(x_vars, self.A[i]) + self.b[i] == y_vars[i])

        z_vars = [m.continuous_var(lb=-m.infinity, ub=m.infinity, name="z(%d,%d)"%(self.layer, i)) for i in range(self.out_dim)]

        for i in range(self.out_dim):
            m.add_constraint(self.sigma[i] * z_vars[i] == self.alpha[i] * y_vars[i] - self.alpha[i] * self.mu[i] + self.sigma[i] * self.gamma[i])

        i_vars = [m.binary_var(name="i(%d,%d)"%(self.layer, i)) for i in range(self.out_dim)]
        v_vars = [m.integer_var(-1, 1, "v(%d,%d)"%(self.layer, i)) for i in range(self.out_dim)]
        
        for i in range(self.out_dim):
            m.add_indicator(i_vars[i], z_vars[i] >= 0, 1)
            m.add_indicator(i_vars[i], z_vars[i] <= 0, 0)
            
            m.add_if_then(i_vars[i] == 1, v_vars[i] == 1)
            m.add_if_then(i_vars[i] == 0, v_vars[i] == -1)

        return v_vars

class Output:
    def __init__(self, in_dim, out_dim, weights, layer):
        self.layer = layer
        self.A = weights[0]
        self.b = weights[1]
    
        self.in_dim = in_dim
        self.out_dim = out_dim

        assert self.A.shape == (out_dim, in_dim)
        assert self.b.shape == (out_dim,)
        
    def evaluate(self, x):

        assert x.shape == (self.in_dim,)

        y = self.A @ x + self.b

        assert y.shape == (self.out_dim,)

        return np.argmax(y)

    def to_cnf(self, x_vars, cnf):
        d_vars = [[cnf.nv + 1 + j + (j*i) for j in range(self.out_dim)] for i in range(self.out_dim)]
        cnf.nv += self.out_dim ** 2

        for j in range(self.out_dim):
            for i in range(self.out_dim):
                E_ij = np.ceil((self.b[i] - self.b[j] + np.sum(self.A[i]) - np.sum(self.A[j])) / 2)

                lits = [x if (ai > 0 and aj < 0) else -x for (x, ai, aj) in zip(x_vars, self.A[i], self.A[j]) if not (np.sign(ai) == np.sign(aj))]

                ale = CardEnc.atleast(lits, np.ceil(E_ij / 2.0), cnf.nv, EncType.seqcounter)
            
                r = ale.nv # TODO: Check assumption that this is "r(n, D)" of AAAI paper

                cnf.extend(ale.clauses)
                cnf.append( [-r,  d_vars[i][j]] )
                cnf.append( [ r, -d_vars[i][j]] )

        return d_vars

    def to_ip(self, x_vars, m):
        y_vars = [m.integer_var(lb=-m.infinity, ub=m.infinity, name="i(%d,%d)"%(self.layer, i)) for i in range(self.out_dim)]

        for i in range(self.out_dim):
            m.add_constraint(np.dot(x_vars, self.A[i]) + np.ceil(self.b[i]) == y_vars[i])

        return y_vars
    
    def to_mip(self, x_vars, m):
        y_vars = [m.continuous_var(lb=-m.infinity, ub=m.infinity, name="y(%d,%d)"%(self.layer, i)) for i in range(self.out_dim)]

        for i in range(self.out_dim):
            m.add_constraint(np.dot(x_vars, self.A[i]) + self.b[i] == y_vars[i])

        return y_vars

# Test on pre-trained binarized NN weights
def load_weight(name):
    return np.load("binary_net/"+name+".npy")

w1 = list(map(load_weight, ["fc1.weight", "fc1.bias", "bn1.weight", "bn1.running_mean", "bn1.running_var", "bn1.bias"]))
w2 = list(map(load_weight, ["fc2.weight", "fc2.bias", "bn2.weight", "bn2.running_mean", "bn2.running_var", "bn2.bias"]))
w3 = list(map(load_weight, ["fc3.weight", "fc3.bias"]))

dim1 = (28*28, 200)
dim2 = (200, 100)
dim3 = (100, 10)

def random_weights(dim):
    A = np.sign(np.random.rand(dim[1], dim[0]) - 0.5)
    b = (np.random.rand(dim[1]) - 0.5) * 10
    alpha = (np.random.rand(dim[1]) - 0.5) * 10
    mu = (np.random.rand(dim[1]) - 0.5) * 10
    sigma = (np.random.rand(dim[1]) - 0.5) * 10
    gamma = (np.random.rand(dim[1]) - 0.5) * 10
    return [A,b,alpha,mu,sigma,gamma]

b1 = Block(*dim1, w1, 0)   #random_weights(dim1))
b2 = Block(*dim2, w2, 1)   #random_weights(dim2))
out = Output(*dim3, w3, 2) #random_weights(dim3)[:2])

correct = 0
for x,y in zip(X_test, Y_test):
    x_flat = x.flatten()
    if out.evaluate(b2.evaluate(b1.evaluate(x_flat))) == y:
        correct += 1
print("Testing network: %.2f%% correct on test set" % (100*correct/len(Y_test)))

m = model.Model("MIP model")
input_vars = np.array([m.integer_var(name="x%d" % i) for i in range(dim1[0])])

## CNF encoding currently explodes in size
# X_test = np.where(X_test>0.5, 1, 0) # binarize input
# solver = Glucose4()
# formula = CNF()
# x_vars = list(range(1,28*28+1))
# formula.nv = len(x_vars)
# t1 = b1.to_cnf(x_vars, formula)
# t2 = b2.to_cnf(t1, formula)
# o = out.to_cnf(t2, formula)
# solver.append_formula(formula)
# print(solver.nof_clauses, solver.nof_vars)
# solver.solve()

# Construct MIP model by chaining layers
# Alternatively to_ip for pure IP model
t1 = b1.to_mip(input_vars, m)
t2 = b2.to_mip(t1, m)
o = out.to_mip(t2, m)

correct = 0
test_n = 10
for j in range(test_n):
    # Set input variable values in MIP model
    for (i,x) in enumerate(X_test[j].flatten()):
        input_vars[i].set_lb(x)
        input_vars[i].set_ub(x)

    m.solve()

    if j == 0:
        m.print_information()
        print(m.get_solve_status())
        print(m.get_solve_details())

    mip_res = np.argmax([v.solution_value for v in o])
    actual  = Y_test[j]
    print("MIP model: %d\t Actual: %d" % (mip_res, actual))

    if mip_res == actual:
        correct += 1

    # Reset input variable values
    for (i,x) in enumerate(X_test[j].flatten()):
        input_vars[i].set_lb(0)
        input_vars[i].set_ub(1)

print("Testing model: %d/%d correct" % (correct,test_n))