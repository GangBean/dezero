import numpy as np

class Variable:
    '''
    변수를 나타내는 클래스.
    '''
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.grad_fn = None

    def set_grad_fn(self, func):
        self.grad_fn = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        funcs = [self.grad_fn]
        while funcs:
            f = funcs.pop()
            x, y = f.variables()
            x.grad = f.backward(y.grad)

            if x.grad_fn:
                funcs.append(x.grad_fn)