import unittest
import numpy as np

from square import Square

class SquareTest(unittest.TestCase):
    def 입력값_제곱_출력(self):
        x = [1, 2, 3, np.ndarray([1,2,3])]
        f = Square
        y = f(x)
        self.assertEqual(y, [1, 4, 9, np.ndarray([1,4,9])])

if __name__ == '__main__':
    unittest.main()