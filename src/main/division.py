from .function import Function

class Division(Function):
    def __init__(self):
        pass

    def forward(self, x0, x1):
        return x0 / x1
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1

def division(x0, x1):
    return Division()(x0, x1)
