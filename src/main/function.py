import numpy as np
from variable import Variable

class Function:
    def __init__(self, calculation):
        self.forward = calculation
    '''
    함수를 나타내는 Base 클래스.
    특정 함수를 구현하기 위해선 해당 클래스를 상속받아야 한다.
    '''
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(np.array(y))
        output.set_grad_fn(self)

        self.input = input
        self.output = output
        return output
    
    '''
    입력값에 대한 함수값을 계산하는 메소드.
    필수로 구현해야 한다.
    '''
    def forward(self, x):
        raise NotImplementedError()
    
    '''
    입력된 선행 노드의 grad값에 해당하는 grad를 계산하는 매소드.
    '''
    def backward(self, gy):
        raise NotImplementedError()
    
    '''
    함수의 미분값을 구하는 메소드.
    중앙차분을 통해 근사한다.
    '''
    def diff(self, x, eps=1e-4):
        x0 = Variable(np.array(x.data - eps))
        x1 = Variable(np.array(x.data + eps))
        y0 = self.__call__(x0)
        y1 = self.__call__(x1)
        return (y1.data - y0.data) / (2 * eps)
    
    def variables(self):
        return self.input, self.output