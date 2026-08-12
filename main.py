from calendar import c
import math
import numpy as np
import matplotlib.pyplot as plt
from value import *
from digraph import *


def f(x):
    return (3 * x**2) - (4 * x) + 5


# print((L2 - L1) / h)
# visualize(draw_bot(L))


# inputs x1, x2
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")
# weights w1, w2
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")
# bias of the neuron
b = Value(6.8813735870195432, label="b")

x1w1 = x1 * w1
x1w1.label = "x1*w1"

x2w2 = x2 * w2
x2w2.label = "x2*w2"

x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = "x1*w1 + x2*w2"
n = x1w1x2w2 + b
n.label = "n"

o = n.tanh()

o.backward()
visualize(draw_bot(o))
