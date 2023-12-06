import numpy as np

class Variable:
    '''
    변수를 나타내는 클래스.
    '''
    def __init__(self, data):
        self.data = self.__ndarray_typed__(data)
        self.grad = None
        self.grad_fn = None
        self.generation = 0
    
    def set_grad_fn(self, func):
        self.grad_fn = func
        self.generation = func.generation + 1
    
    def clear_grad(self):
        self.grad = None

    def backward(self):
        self.__init_empty_grad_with_ones__()
        self.__propagate_grads__()
    
    def __ndarray_typed__(self, data):
        if data and not isinstance(data, np.ndarray):
            raise TypeError(f"Numpy ndarray타입만 사용 가능합니다: {type(data)}")
        return data
    
    def __init_empty_grad_with_ones__(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

    def __propagate_grads__(self):
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
    
    def __add_funcs_and_sort__(self, f, funcs, calculated_funcs):
        if f not in calculated_funcs:
            funcs.append(f)
            calculated_funcs.add(f)
            funcs.sort(key=lambda x: x.generation)