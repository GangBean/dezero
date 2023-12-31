import unittest
import numpy as np
from src.main.variable import Variable
from src.main.config import Config

from src.main.square import square

class SquareTest(unittest.TestCase):
    def test_입력값_제곱_출력(self):
        x = Variable(np.array(3))
        y = square(x)
        self.assertEqual(y.data, 9)

    def test_출력_variable의_backward_호출시_모든변수의_그레디언트가_계산됩니다(self):
        Config.enable_backprop = True
        x = Variable(np.array(10))
        y = square(x)
        z = square(y)
        z.backward(retain_grad=True)

        self.assertEqual(z.grad, 1)
        self.assertEqual(y.grad, 200)
        self.assertEqual(x.grad, 4_000)

if __name__ == '__main__':
    unittest.main()