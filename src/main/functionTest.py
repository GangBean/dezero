import numpy as np
import unittest
from function import Function
from variable import Variable
from config import Config
from square import square

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
    
    def test_enable_backprop이_True일때_grad_fn이_저장됩니다(self):
        Config.enable_backprop = True
        x = Variable(np.array(10))
        y = square(square(square(x)))

        self.assertIsNotNone(y.grad_fn)
    
    def test_enable_backprop이_False일때_grad_fn이_저장되지_않습니다(self):
        Config.enable_backprop = False
        x = Variable(np.array(10))
        y = square(square(square(x)))

        self.assertIsNone(y.grad_fn)

if __name__ == '__main__':
    unittest.main()