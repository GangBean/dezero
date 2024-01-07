import unittest
import numpy as np
from src.main.variable import Variable
from src.main.square import square

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
        y = square(a) + square(a)
        y.backward()

        self.assertAlmostEqual(x.grad, 64.)
    
    def test_retain이_false면_중간미분값이_저장되지않습니다(self):
        x0 = Variable(np.array(1.0))
        x1 = Variable(np.array(1.0))
        t = x0 + x1
        y = x0 + t
        y.backward()

        self.assertIsNone(y.grad)
        self.assertIsNone(t.grad)
    
    def test_retain이_true면_중간미분값이_저장됩니다(self):
        x0 = Variable(np.array(1.0))
        x1 = Variable(np.array(1.0))
        t = x0 + x1
        y = x0 + t
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

    def test_변수는_데이터의_size를_알려줍니다(self):
        x = Variable(np.array([[1,2,3], [4,5,6]]))

        self.assertEqual(x.size, 6)

    def test_변수는_데이터의_dtype을_알려줍니다(self):
        x = Variable(np.array([1, 3, 4]))

        self.assertEqual(x.dtype, np.int32)

    def test_변수에_len함수를사용하면_첫번째차원의_원소수를_알려줍니다(self):
        x = Variable(np.array([1,2,3,4]))

        length = len(x)

        self.assertEqual(length, 4)

    def test_변수를_print하면_보기좋은형태로_출력합니다(self):
        x = Variable(np.array([[1,2,3,4],[5,6,7,8]]), 'x')

        self.assertRegexpMatches(x.__repr__(), r'x: variable\(\[\[1 2 3 4\]\n\s+ \[5 6 7 8\]\]\)')

    def test_변수끼리_플러스연산자를사용하면_add결과를_출력합니다(self):
        a = Variable(np.array(1.0))
        b = Variable(np.array(2.0))

        y = a + b

        self.assertAlmostEqual(y.data, 3.0)

    def test_변수끼리_곱하기연산자를사용하면_multiply결과를_출력합니다(self):
        a = Variable(np.array(1.0))
        b = Variable(np.array(2.0))

        y = a * b

        self.assertAlmostEqual(y.data, 2.0)

    def test_Variable는_다른숫자형변수를_Variabel로_변환해줍니다(self):
        a = Variable.as_variable(1)

        self.assertEqual(type(a), Variable)

    def test_Variable_은_마이너스부호를_붙이면_데이터의부호가_변합니다(self):
        a = Variable(np.array(2.0))
        neg_a = -a

        self.assertEqual(neg_a.data, np.array(-2.0))

    def test_Variable에서_다른Variable을_뺄수있습니다(self):
        a = Variable(np.array(3.0))
        b = Variable(np.array(1.0))
        c = a - b

        self.assertEqual(c.data, np.array(2.0))

    def test_Variable에서_상수를_뺄수있습니다(self):
        a = Variable(np.array(3.0))
        c = a - 1

        self.assertEqual(c.data, np.array(2.0))
    
    def test_Variable를_상수에서_뺄수있습니다(self):
        a = Variable(np.array(3.0))
        c = 5 - a

        self.assertEqual(c.data, np.array(2.0))
    
    def test_Variable을_Variable로_나눌수있습니다(self):
        a = Variable(np.array(3.0))
        b = Variable(np.array(3.0))
        c = a / b

        self.assertEqual(c.data, np.array(1.0))
    
    def test_상수를_Variable로_나눌수있습니다(self):
        a = np.array(3.0)
        b = Variable(np.array(3.0))
        c = a / b

        self.assertEqual(c.data, np.array(1.0))

if __name__ == '__main__':
    unittest.main()