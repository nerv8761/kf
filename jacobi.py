import sympy
import numpy as np
import scipy.linalg
from copy import deepcopy
from threading import Lock
import math
import matplotlib.pyplot as plt
x0,x1,x2,x3,x4,ret0,ret1,ret2,ret3,ret4 = sympy.symbols("x0 x1 x2 x3 x4 ret0 ret1 ret2 ret3 ret4")
ret0 = x0 + (x2 / x4) * ((sympy.sin(x3 + x4 * 0.05)) - sympy.sin(x3))
ret1 = x1 + (x2 / x4) * ((-sympy.cos(x3 + x4 * 0.05)) + sympy.cos(x3))
ret2 = x2
ret3 = x3 + 0.05 * x4
ret4 = x4
funcs = sympy.Matrix([ret0,ret1,ret2,ret3,ret4])
args = sympy.Matrix([x0,x1,x2,x3,x4])
res = funcs.jacobian(args)
a=np.array([[1,2,3],
           [2,3,4]])
b=np.array([4.4,0,0,5])

p = 100
x_p = np.zeros((p,5))
for i in range(0,p):
    x_p[i] = np.random.randn(5)



print(res)

