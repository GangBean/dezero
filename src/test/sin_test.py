import unittest
import numpy as np
from src.main.sin import sin
from src.main.variable import Variable

class SinTest(unittest.TestCase):
    def test_sin함수값을_계산합니다(self):
        x = Variable(np.array(np.pi/4))
        y = sin(x)

        self.assertAlmostEqual(y.data, 0.707106781)
    
    def test_sin함수의_미분값을_계산합니다(self):
        x = Variable(np.array(np.pi/4))
        y = sin(x)
        y.backward()

        self.assertAlmostEqual(x.grad, 0.707106781)