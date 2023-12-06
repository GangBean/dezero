import numpy as np
import unittest
from function import Function
from variable import Variable

class FunctionTest(unittest.TestCase):
    def test_forward_해당하는_계산결과를_리턴합니다(self):
        x = Variable(np.array(10))
        f = Function(lambda *xs: [x ** 2 for x in xs])
        y = f(x)
        self.assertEqual(y.data, 100)
    
    def test_diff_미분결과를_리턴합니다(self):
        x = Variable(np.array(10))
        f = Function(lambda *xs: [x ** 2 for x in xs])
        y = f.diff(x)
        self.assertAlmostEqual(y[0], 20)
    
    def test_forward시_결과변수는_연산함수를_저장합니다(self):
        x = Variable(np.array(10))
        f = Function(lambda *xs: [x ** 2 for x in xs])
        y = f(x)
        self.assertEqual(y.grad_fn, f)

if __name__ == '__main__':
    unittest.main()