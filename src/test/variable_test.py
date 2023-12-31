import unittest
import numpy as np
from src.main.variable import Variable
from src.main.square import square
from src.main.add import add

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

    def test_다중계산그래프_역전파도_정상적으로_계산됩니다(self):
        x = Variable(np.array(2.))
        a = square(x)
        y = add(square(a), square(a))
        y.backward()

        self.assertAlmostEqual(x.grad, 64.)
    
    def test_retain이_false면_중간미분값이_저장되지않습니다(self):
        x0 = Variable(np.array(1.0))
        x1 = Variable(np.array(1.0))
        t = add(x0, x1)
        y = add(x0, t)
        y.backward()

        self.assertIsNone(y.grad)
        self.assertIsNone(t.grad)
    
    def test_retain이_true면_중간미분값이_저장됩니다(self):
        x0 = Variable(np.array(1.0))
        x1 = Variable(np.array(1.0))
        t = add(x0, x1)
        y = add(x0, t)
        y.backward(retain_grad=True)

        self.assertIsNotNone(y.grad)
        self.assertIsNotNone(t.grad)

    def test_변수는_이름을_알려줍니다(self):
        x = Variable(np.array(1.0), 'x')
        y = Variable(np.array(1.9), 'y')

        self.assertEqual(x.name, 'x')
        self.assertEqual(y.name, 'y')

    def test_변수는_이름을_지정하지않으면_None을_갖습니다(self):
        x = Variable(np.array(1.0))

        self.assertIsNone(x.name)

    def test_변수는_데이터의_shape을_알려줍니다(self):
        x = Variable(np.array([1.0]), 'x')

        self.assertEqual(x.shape, (1,))
    
    def test_변수는_데이터의_ndim을_알려줍니다(self):
        x = Variable(np.array([[[1, 2], [3, 4]]]))

        self.assertEqual(x.ndim, 3)

if __name__ == '__main__':
    unittest.main()