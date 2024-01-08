import numpy as np
from .function import Function

class Sin(Function):
    def __init__(self):
        pass

    def forward(self, x):
        return np.sin(x)
    
    def backward(self, gy):
        return gy * np.cos(self.inputs[0].data)
    
def sin(x):
    return Sin()(x)
