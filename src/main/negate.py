from .function import Function

class Negate(Function):
    def __init__(self):
        pass
    
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy

def negate(x):
    return Negate()(x)
