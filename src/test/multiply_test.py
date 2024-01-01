import numpy as np
import unittest
from src.main.variable import Variable
from src.main.multiply import multiply

class MultiplyTest(unittest.TestCase):
    def test_multiply를통해_variable_2개의_곱을_갖는_variable을_구할수_있습니다(self):
        a = Variable(np.array(1.0))
        b = Variable(np.array(2.0))

        y = multiply(a, b)

        self.assertAlmostEqual(y.data, 2.0)
    
    def test_Multiply_backward를_통해_최종결과에대한_입력값별_기울기를_구할수있습니다(self):
        a = Variable(np.array(1.0))
        b = Variable(np.array(2.0))

        y = multiply(a, b)

        y.backward()

        self.assertEqual(a.grad, 2.0)
        self.assertEqual(b.grad, 1.0)
