from .function import Function

class Subtract(Function):
    def __init__(self):
        pass

    def forward(self, x0, x1):
        return x0 - x1

    def backward(self, gy):
        return gy, -gy

def subtract(x0, x1):
    return Subtract()(x0, x1)
