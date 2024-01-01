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
