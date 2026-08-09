class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"fValue(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")
        return out

    def __subtract__(self, other):
        out = Value(self.data - other.data, (self, other), "-")
        return out

    def __div__(self, other):
        out = Value(self.data / other.data, (self, other), "/")
        return out
