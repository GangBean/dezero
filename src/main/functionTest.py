import numpy as np
import unittest
from function import Function
from variable import Variable

class FunctionTest(unittest.TestCase):
    def Function_call_은_계산결과를_리턴합니다(self):
        x = Variable(np.ndarray(10))
        f = Function
        y = f(x)
        print(y)
        self.assertEqual(y, 100)

if __name__ == '__main__':
    unittest.main()