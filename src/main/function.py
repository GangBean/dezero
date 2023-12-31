import numpy as np
import weakref
from .config import Config

from .variable import Variable

class Function:
    def __init__(self, calculation):
        self.forward = calculation


    '''
    함수를 나타내는 Base 클래스.
    특정 함수를 구현하기 위해선 해당 클래스를 상속받아야 한다.
    '''
    def __call__(self, *inputs):
        inputs = [Variable.as_variable(input) for input in inputs]
        
        outputs = self.__forward_results(inputs)
        self.__prepare_back_propagation(inputs, outputs)

        return outputs[0] if len(outputs) == 1 else outputs
    

    '''
    입력값에 대한 함수값을 계산하는 메소드.
    필수로 구현해야 한다.
    '''
    def forward(self, *xs):
        raise NotImplementedError()
    

    '''
    입력된 선행 노드의 grad값에 해당하는 grad를 계산하는 매소드.
    '''
    def backward(self, gy):
        raise NotImplementedError()
    

    '''
    함수의 미분값을 구하는 메소드.
    중앙차분을 통해 근사한다.
    입력:
        xs: List<Variable>
    출력:
        List<Float>
    '''
    def diff(self, *xs, eps=1e-4):
        xs0 = [Variable(np.array(x.data - eps)) for x in xs]
        xs1 = [Variable(np.array(x.data + eps)) for x in xs]
        ys0 = self.__call__(*xs0)
        ys1 = self.__call__(*xs1)

        if not isinstance(ys0, list):
            ys0 = [ys0]
        if not isinstance(ys1, list):
            ys1 = [ys1]
        
        ret = []
        for y0, y1 in zip(ys0, ys1):
            ret.append((y1.data - y0.data) / (2 * eps))
        return ret[0] if len(ret) == 1 else ret
    

    def output_grads(self):
        return (output().grad for output in self.outputs)
    

    def __forward_results(self, inputs):
        xs = [x.data for x in inputs]
        ys = Function.__tupled(self.forward(*xs))
        
        return [Variable.as_variable(y) for y in ys]
    

    @staticmethod
    def __tupled(input):
        if isinstance(input, tuple):
            return input
        return (input,)
    

    def __prepare_back_propagation(self, inputs, outputs):
        if not Config.enable_backprop:
            return
        
        self.generation = max((x.generation for x in inputs))
        self.inputs = inputs
        self.__set_outputs_grad_fn(outputs)
        self.outputs = [weakref.ref(output) for output in outputs]
    
    
    def __set_outputs_grad_fn(self, outputs):
        for output in outputs:
            output.set_grad_fn(self)