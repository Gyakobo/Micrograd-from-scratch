from calendar import c
import math
import numpy as np
import matplotlib.pyplot as plt
from value import *
from digraph import *


def f(x):
    return (3 * x**2) - (4 * x) + 5


h = 0.001

a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
e = a * b
e.label = "e"
d = e + c
d.label = "d"
f = Value(-2.0, label="f")
L = d * f
L.label = "L"
L1 = L.data  # This is a value node


a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
e = a * b
e.label = "e"
d = e + c
d.label = "d"
f = Value(-2.0, label="f")
L = d * f
L.label = "L"
L2 = L.data + h  # This is a value node

L.grad = 1
f.grad = 4
d.grad = -2

print((L2 - L1) / h)
visualize(draw_bot(L))
