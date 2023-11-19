import numpy as np
from variable import Variable

class Function:
    '''
    함수를 나타내는 Base 클래스.
    특정 함수를 구현하기 위해선 해당 클래스를 상속받아야 한다.
    '''
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output
    
    def forward(self, x):
        raise NotImplementedError()
