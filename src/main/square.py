from function import Function

class Square(Function):
    def __init__(self):
        pass

    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        return 2 * self.input.data * gy