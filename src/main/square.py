from function import Function

class Square(Function):
    def __init__(self):
        pass
    
    def forward(self, x):
        return x ** 2