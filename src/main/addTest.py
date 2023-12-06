import unittest
from variable import Variable
import numpy as np
from add import add

class AddTest(unittest.TestCase):
    def test_초기화를하면_같은변수를써도_미분값이정상계산됩니다(self):
        x = Variable(np.array(10.))
        y = add(x, x)
        y.backward()
        self.assertEqual(x.grad, 2.)

        x.clear_grad()
        self.assertEqual(x.grad, None)


if __name__ == '__main__':
    unittest.main()