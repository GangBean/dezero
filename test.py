import unittest

if __name__ == '__main__':
    suite = unittest.TestLoader().discover('./src/test', pattern='*_test.py')

    unittest.TextTestRunner().run(suite)
