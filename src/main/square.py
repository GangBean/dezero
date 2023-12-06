from function import Function

class Square(Function):
    def __init__(self):
        pass

    def forward(self, *xs):
        return [x ** 2 for x in xs]
    
    def backward(self, gy):
        return 2 * self.inputs[0].data * gy

def square(x):
    return Square()(x)