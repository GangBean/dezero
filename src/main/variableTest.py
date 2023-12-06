import unittest
import numpy as np
from variable import Variable

class VariableTest(unittest.TestCase):
    def test_변수는_ndarray타입을_사용합니다(self):
        x = Variable(np.array(10))

        self.assertEqual(type(x.data), np.ndarray)

    def test_변수가_ndarray타입이아닐경우_TypeError가_발생합니다(self):
        data = 10
        self.assertNotIsInstance(data, np.ndarray)

        with self.assertRaises(TypeError) as context:
            x = Variable(data)
        self.assertRegex(str(context.exception), "Numpy ndarray타입만 사용 가능합니다")

if __name__ == '__main__':
    unittest.main()