from .function import Function

class Multiply(Function):
    def __init__(self):
        pass
    
    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0

def multiply(x0, x1):
    return Multiply()(x0, x1)