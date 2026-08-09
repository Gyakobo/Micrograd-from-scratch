from calendar import c
import math
import numpy as np
import matplotlib.pyplot as plt
from value import *
from digraph import *


def f(x):
    return (3 * x**2) - (4 * x) + 5


a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(1.0, label="c")

e = a * b
e.label = "e"

d = e + c
d.label = "d"

f = Value(-2.0, label="f")
L = d * f
L.label = "L"

visualize(draw_bot(L))
