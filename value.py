class Value:
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"fValue(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data)
        return out

    def __mul__(self, other):
        if type(other) == "<class 'value.Value'>":
            out = Value(self.data * other.data)
        elif type(other) == "<class 'int'>":
            out = Value(self.data * other)
        return out
