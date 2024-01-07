import numpy as np

class Variable:
    '''
    변수를 나타내는 클래스.
    '''
    def __init__(self, data, name:str = None):
        self.data = self.__ndarray_typed(data)
        self.grad = None
        self.grad_fn = None
        self.generation = 0
        self.name = name
        self.__array_priority__ = 1
    
    def set_grad_fn(self, func):
        self.grad_fn = func
        self.generation = func.generation + 1
    
    def clear_grad(self):
        self.grad = None

    def backward(self, retain_grad = False):
        self.__init_empty_grad_with_ones__()
        self.__propagate_grads__(retain_grad)
    
    @staticmethod
    def as_variable(data):
        if isinstance(data, Variable):
            return data
        return Variable(Variable.__as_array(data))
    
    @staticmethod
    def __as_array(data):
        if isinstance(data, np.ndarray):
            return data
        return np.array(data)

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        name = 'None' if self.name is None else self.name
        if self.data is None:
            return name + ': variable(None)'
        p = str(self.data).replace('\n', '\n' + (' ' * 9))
        return name + ': variable(' + p + ')'
    
    def __add__(self, other):
        from .add import add
        return add(self, other)
    
    def __radd__(self, other):
        from .add import add
        return add(self, other)
    
    def __mul__(self, other):
        from .multiply import multiply
        return multiply(self, other)
    
    def __rmul__(self, other):
        from .multiply import multiply
        return multiply(self, other)
    
    def __neg__(self):
        from .negate import negate
        return negate(self)
    
    def __sub__(self, other):
        from .substract import subtract
        return subtract(self, other)
    
    def __rsub__(self, other):
        from .substract import subtract
        return subtract(other, self)
    
    def __truediv__(self, other):
        from .division import division
        return division(self, other)
    
    def __rtruediv__(self, other):
        from .division import division
        return division(other, self)
    
    def __ndarray_typed(self, data):
        if self.__is_not_valid_data(data):
            raise TypeError(f"Numpy ndarray타입만 사용 가능합니다: {type(data)}")
        return data
    
    def __is_numpy_array(self, data):
        return isinstance(data, np.ndarray)
    
    def __is_not_valid_data(self, data):
        if data is None:
            return True
        if self.__is_numpy_array(data):
            return False
        if isinstance(data, list):
            return any(not self.__is_numpy_array(item) for item in data)
        return True
    
    def __init_empty_grad_with_ones__(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

    def __propagate_grads__(self, retain_grad):
        funcs = []
        calculated_funcs = set()

        self.__add_funcs_and_sort__(self.grad_fn, funcs, calculated_funcs)
        
        while funcs:
            f = funcs.pop()
            gys = f.output_grads()
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = 0
                x.grad = x.grad + gx # np.array 는 += 연산시 inplace 연산이 발생함. 이를 막기 위해 명시적으로 풀어써줌

                if x.grad_fn is not None:
                    self.__add_funcs_and_sort__(x.grad_fn, funcs, calculated_funcs)
        
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None
    
    def __add_funcs_and_sort__(self, f, funcs, calculated_funcs):
        if f not in calculated_funcs:
            funcs.append(f)
            calculated_funcs.add(f)
            funcs.sort(key=lambda x: x.generation)