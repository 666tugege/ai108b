import gd2 as gd
import numpy as np
from numpy.linalg import norm
    
A = np.array([[0, 1, 1], 
             [1, 0, 1], 
             [1, 1, 1]])
B = np.array([0.0, 0, 1]).transpose()

def sig(x):
    s = 1 / (1 + np.exp(-x))
    return s

def f(p):
    X = p
    Y = np.dot(A, X)
    return norm(Y - B)

p = np.array([0.0, 0, 0])
p = gd.gradientDescendent(f, p)
print("\n==================================================================================\n")

print("p = {}".format(p))

weights = np.array([p[0], p[1]])
bias = p[2]

inputArr = np.array([[0, 0], 
             [0, 1],
             [1, 0], 
             [1, 1]])
B = np.dot(inputArr, weights) + bias
print("Inputs: \n{}".format(inputArr))
print("\u03A3(in⋅w) + b = {}".format(B))
print("sig(\u03A3) = {}".format(sig(B)))

ans = []
for i in B:
    ans.append(0) if sig(i) <= 0.5 else ans.append(1)
    # ans.append(0) if sig(i) <= 0.6 else ans.append(1)
print("Output = {}".format(ans))
print("\n==================================================================================\n")

print("weights: {}".format(weights))
print("bias: {}".format(bias))

print("--------------------------------")

print("\tInputs  |  Outputs")
for i in range(len(ans)):
    print("\t{}\t\t{}".format(inputArr[i], ans[i]))
