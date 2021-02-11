import unittest
from sampleFunctions import *
from tqdm import tqdm

import scipy.integrate as integrate
import matplotlib.pyplot as plt
import tfunctions
import assignment2
from assignment3 import Assignment3


class TestAssignment3(unittest.TestCase):

    def test_integrate_float32(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        r = ass3.integrate(f1, -1, 1, 10)

        self.assertEqual(r.dtype, np.float32)

    """def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = strong_oscilations()
        r = ass3.integrate(f1, 0.09, 10, 20)
        true_result = -7.78662 * 10 ** 33
        print(abs((r - true_result) / true_result))
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))"""

    def test_integrate_1(self):
        self.tfunc_integrate("f(x) = 5", tfunctions.f1, -10, 10, 1000)

    def test_integrate_2(self):
        function = np.poly1d([1, 0, 0, -2, 0, 5])
        self.tfunc_integrate("x^5 - 2x^2 + 5", function, 1, 10, 2000)

    def test_integrate_3(self):
        self.tfunc_integrate("sin(x^2)", tfunctions.f3, -10, 10, 3500)

    def test_integrate_4(self):
        function = np.poly1d([1, 0, 0])
        self.tfunc_integrate("x^2", function, -100, 100, 1000)

    def test_integrate_5(self):
        self.tfunc_integrate("ln(np.ln(x))", tfunctions.f9, 2, 50, 1000)

    def test_areabetween_1(self):
        function2 = lambda x: np.poly1d([1, 0, 0])(x - 10)
        self.tfunc_areabetween("f1(x) = 5, f2(x) = (x - 10)^2", tfunctions.f1, function2)

    def test_areabetween_2_nan(self):
        function2 = lambda x: np.poly1d([1, 0, 0])(x + 10)
        self.tfunc_areabetween("f1(x) = 5, f2(x) = (x + 10)^2", tfunctions.f1, function2)

    def test_areabetween_3(self):
        function1 = lambda x: np.poly1d([1, 0, 0])(x - 10)
        function2 = lambda x: np.poly1d([-1, 0, 10])(x - 10)
        self.tfunc_areabetween("f1(x) = (x - 10)^2, f2(x) = -(x - 10)^2 + 10", function1, function2)

    def test_areabetween_4(self):
        function1 = lambda x: np.sin(x+0.2) + 5
        function2 = lambda x: np.sin(x) + 5
        self.tfunc_areabetween("f1(x) = sin(x+0.2), f2(x) = sin(x)", function1, function2)

    def tfunc_integrate(self, func_name: str, f1: callable, s: float, to: float, n: int, maxerr: float=0.001, draw=False):
        if (draw):
            p_x = np.arange(s * 1.1, to * 1.1, 0.1)
            p_y = f1(p_x)
            plt.plot(p_x, p_y)
            plt.show()


        expect_area = integrate.quad(lambda x: f1(x), s, to)
        f1 = RESTRICT_INVOCATIONS(n)(f1)
        ass3 = Assignment3()
        r = ass3.integrate(f1, s, to, n)
        # print(integrate.simps(f1(points), points))
        # print(f"{expect_aera} {r}")
        self.assertEqual(r.dtype, np.float32)
        self.assertGreaterEqual(maxerr, abs((r - expect_area[0]) / expect_area[0]), f"{func_name}, {r}, {expect_area}")

    def tfunc_areabetween(self, func_name: str, f1: callable, f2: callable, are_intersects: bool=True, maxerr: float=0.001, draw=False):
        if (draw):
            p_x = np.arange(0, 100)
            p_y = f1(p_x) - f2(p_x)
            plt.plot(p_x, p_y)
            plt.show()

        ass3 = Assignment3()
        r = ass3.areabetween(f1, f2)

        intersections = assignment2.Assignment2().intersections(f1, f2, 1, 100)
        if((not are_intersects)):
            self.assertTrue(np.isnan(r))
        if(len(intersections) < 2):
            self.assertTrue(np.isnan(r))
        else:
            expect_area = 0
            for i in range(0, len(intersections) - 1):
                expect_area = expect_area + abs(integrate.quad(lambda x: f1(x) - f2(x), intersections[i], intersections[i + 1])[0])
            # print(f"{expect_aera} {r}")
            self.assertEqual(r.dtype, np.float32)
            self.assertGreaterEqual(maxerr, abs((r - expect_area) / expect_area),
                                    f"\n{func_name}\n\t relative: {abs((r - expect_area) / expect_area)}\n\t actual {r}\n\t expect {expect_area}")

if __name__ == "__main__":
    unittest.main()
