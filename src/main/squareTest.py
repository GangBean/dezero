import unittest
import numpy as np
from variable import Variable

from square import square

class SquareTest(unittest.TestCase):
    def test_입력값_제곱_출력(self):
        x = Variable(3)
        y = square(x)
        self.assertEqual(y.data, 9)

if __name__ == '__main__':
    unittest.main()