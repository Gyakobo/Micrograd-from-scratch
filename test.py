import math


class Value:
    def __init__(self, data, children=(), _op="", label=""):
        # Data
        self.data = data

        # Syntax Tree
        self._prev = children
        self._op = _op
        self.label = label

        # Gradient Descent
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self):
        return f"fValue(data={self.data}, grad={self.grad}, op={self._op}, label={self.label})"

    def show_children(self):
        print(self._prev)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), _op="+")

        def _backward():
            self.grad += 1.0 * out.grad  # local derivative
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), _op="*")

        def _backward():
            self.grad += other.data * out.grad  # chain rule
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data**other.data, (self, other), _op=f"**{other}")
        return out

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * Value**-1

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


a = Value(1)
a.label = "a"

b = Value(2)
b.label = "b"

c = Value(3)
c.label = "c"

d = Value(4)
d.label = "d"

L = (a + b) * c + d
L.label = "L"

L.backward()
