class Value:
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"fValue(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data)
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data)
        return out

    def __radd__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data)
        return out

    def __rmul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data)
        return out

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data**other.data)
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data)
        return out

    def __rtruediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(other.data / self.data)
        return out

    def __neg__(self):
        return self * -1.0


obj1 = Value(2)
obj2 = Value(3)

print(-obj1)
