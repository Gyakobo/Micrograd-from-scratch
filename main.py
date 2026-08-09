import math
import numpy as np
import matplotlib.pyplot as plt
from value import *


def f(x):
    return (3 * x**2) - (4 * x) + 5


xs = np.arange(-5, 5, 0.25)
ys = f(xs)

"""
plt.plot(xs, ys)
plt.show()
"""

h = 0.001
x = 3.0

a = Value(2.0)
b = Value(-3.0)

print(a)
