import math
import random
import numpy as np

np.random.seed(1337)
random.seed(1337)


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
        # return f"fValue(data={self.data}, grad={self.grad}, op={self._op}, label={self.label})"
        return f"fValue(data={self.data})"

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

    def __rsub__(self, other):
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


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]  # wi
        self.b = Value(random.uniform(-1, 1))  # b

    def __call__(self, x: list):
        # wi * xi + b
        activation = sum(wi * xi for (wi, xi) in zip(self.w, x)) + self.b
        out = activation.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(
        self, nin, nout
    ):  # n of dimensions(input per neuron), quantity of neurons in a single layer
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs  # for convenience

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(
        self, nin, nouts
    ):  # n of dimensions(input per neuron), lists of nouts(sizes of all the layers in our MLP)
        sz = [nin] + nouts  # [inputs per neuron], []
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


nn = MLP(3, [4, 4, 1])

xs = [
    [2, 3, -1],
    [3, -1, 0.5],
    [0.5, 1, 1],
    [1, 1, -1],
]
ys = [1, -1, -1, 1]

for _ in range(4):
    # Forward pass
    ypred = [nn(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))
    print("loss:", loss)

    # Backward pass
    loss.backward()

    # Update
    print(nn.layers[0].neurons[0].w[0].data)
    for p in nn.parameters():
        p.data += -0.01 * p.grad
    print(nn.layers[0].neurons[0].w[0].data)
