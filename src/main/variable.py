class Variable:
    '''
    변수를 나타내는 클래스.
    '''
    def __init__(self, data):
        self.data = data
        self.grad = None