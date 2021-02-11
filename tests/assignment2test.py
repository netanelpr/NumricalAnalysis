import unittest
from sampleFunctions import *
from tqdm import tqdm
import tfunctions
import matplotlib.pyplot as plt
from assignment2 import Assignment2

class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_sin(self):

        f1 = lambda x: np.sin(x)
        f2 = lambda x: 0

        self.tfunc("sin", f1, f2, -10, 10, 0.001)

    def test_exp(self):

        f1 = lambda x: np.exp(x)
        f2 = lambda x: 0

        self.tfunc("exp", f1, f2, -10, 10, 0.001)

    def test_const(self):

        f2 = lambda x: 0

        self.tfunc("t1", tfunctions.f1, f2, -1000, 1000, 0.001, 0)

    def test_2(self):

        f2 = lambda x: 0

        self.tfunc("t2", tfunctions.f2, f2, -5, 5, 0.001, 0)

    def test_3(self):

        f2 = lambda x: 0

        self.tfunc("t3", tfunctions.f3, f2, -4.1, 4.1, 0.001, 11)

    def test_4(self):

        f2 = lambda x: 0

        self.tfunc("t4", tfunctions.f4, f2, -1, 1, 0.001, 0)

    def test_5(self):

        f2 = lambda x: 0

        self.tfunc("t5", tfunctions.f5, f2, -100, 100, 0.001, 1)

    def test_6(self):

        f2 = lambda x: 0

        self.tfunc("t6", tfunctions.f6, f2, -4, 4, 0.001, 2)

    def test_7(self):

        f2 = lambda x: 0

        self.tfunc("t7", tfunctions.f7, f2, 0.1, 0.999, 0.001, 0)
        self.tfunc("t7", tfunctions.f7, f2, 1.0001, 10, 0.001, 0)

    def test_8(self):

        f2 = lambda x: 0

        self.tfunc("t8", tfunctions.f8, f2, -4, 4, 0.001, 0)

    def test_9(self):

        f2 = lambda x: 0

        self.tfunc("t9", tfunctions.f9, f2, 2.1, 5, 0.001, 1)

    def test_10(self):

        f2 = lambda x: 0

        self.tfunc("t10", tfunctions.f10, f2, 0.001, 600, 0.001, 3)

    def test_11(self):

        f2 = lambda x: 0

        self.tfunc("t11", tfunctions.f11, f2, -1, -0.01, 0.001, 1)
        self.tfunc("t11", tfunctions.f11, f2, 0.01, 1, 0.001, 2)

    def test_sqrt(self):

        f_poly = np.poly1d([1, 0, 0])
        f1 = lambda x: np.sqrt(f_poly(x))
        f2 = lambda x: 0

        self.tfunc("sqrt", f1, f2, -100, 100, 0.001, 1)

    def test_12(self):

        function = np.poly1d([1, -6, 8, 0])

        f2 = lambda x: 0

        self.tfunc("t12", lambda x: function(x) * np.sin(x), f2, -4, 5, 0.001, 5)

    def test_13(self):

        function = np.poly1d([1, -6, 8, 0])

        f1 = np.poly1d([1, -2, 0, 1])
        f2 = lambda x: x

        self.tfunc("t13", f1, f2, -10, 10, 0.001, 3)

    def tfunc(self, func_name: str, f1: callable, f2: callable, s: float, to: float, maxerr: float, number_of_points=-1,
              draw=False):

        print(func_name)
        if (draw):
            p_x = np.arange(s * 1.1, to * 1.1, 0.1)
            p_y = f1(p_x) - f2(p_x)
            plt.plot(p_x, p_y)
            plt.show()

        ass2 = Assignment2()
        X = ass2.intersections(f1, f2, s, to, maxerr)
        index = 1
        for x in X:
            print(f"{index} {x}")
            index = index + 1
            self.assertGreaterEqual(maxerr, abs(f1(x) - f2(x)))

        if (number_of_points > -1):
            self.assertEqual(number_of_points, index - 1)
        print(f"found {index - 1} points")

if __name__ == "__main__":
    unittest.main()