"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and 
the leftmost intersection points of the two functions. 

The functions for the numeric answers are specified in MOODLE. 


This assignment is more complicated than Assignment1 and Assignment2 because: 
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors. 
    2. You have the freedom to choose how to calculate the area between the two functions. 
    3. The functions may intersect multiple times. Here is an example: 
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately. 
       You should explain why in one of the theoretical questions in MOODLE. 

"""

import numpy as np
import time
import random
import assignment2

class Assignment3:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the integration error. 
        Your secondary objective is minimizing the running time. The assignment
        will be tested on variety of different functions. 
        
        Integration error will be measured compared to the actual value of the 
        definite integral. 
        
        Note: It is forbidden to call f more than n times. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
        """

        def composite_simpson(a: np.float32, b: np.float32) -> np.float32:
            h = (b - a) / np.float32(n)
            sum1_array = [f(a)]
            sum2_array = []
            sum3_array = [f(b)]

            x = a + h
            for i in range(1, n):
                point = np.float32(f(x))

                if(i % 2 == 0):
                    sum3_array.append(point)
                    sum1_array.append(point)
                else:
                    sum2_array.append(point)

                x = x + h

            sum1_array.sort()
            sum2_array.sort()
            sum3_array.sort()
            sum1 = np.sum(sum1_array, dtype=np.float32)
            sum2 = np.float32(4.0) * np.sum(sum2_array, dtype=np.float32)
            sum3 = np.sum(sum3_array, dtype=np.float32)

            return (h * (sum1 + sum3 + sum2) / np.float32(3.0))

        return composite_simpson(np.float32(a), np.float32(b))

    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Finds the area enclosed between two functions. This method finds 
        all intersection points between the two functions to work correctly. 
        
        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        Note, there is no such thing as negative area. 
        
        In order to find the enclosed area the given functions must intersect 
        in at least two points. If the functions do not intersect or intersect 
        in less than two points this function returns NaN.  
        This function may not work correctly if there is infinite number of 
        intersection points. 
        

        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis

        """

        def f(x):
            return f1(x) - f2(x)

        def calcSlope(f, p, delta):
            y_p = f(p)
            y_to = f(p + delta)
            return (y_to - y_p) / delta

        def is_more_intersections(slopes_array):
            if (slopes_array[0] < 0):
                for i in range(1, len(slopes_array)):
                    if (slopes_array[i - 1] < slopes_array[i]):
                        return True
            else:
                for i in range(1, len(slopes_array)):
                    if (slopes_array[i - 1] > slopes_array[i]):
                        return True
            return False

        def find_intersections_left(f1, f2, calcIntersections, at, slopes_delta, jmp):
            slopes = [0, 0, 0]
            next_at = at - jmp
            intersections = calcIntersections(f1, f2, next_at, at, 0.001)
            # print(intersections)
            if (len(intersections) == 0):
                for i in range(0, len(slopes_delta)):
                    delta = slopes_delta[i]
                    slopes[i] = calcSlope(lambda x: f1(x) - f2(x), at, (delta * -1)) * -1
                if (is_more_intersections(slopes)):
                    return find_intersections_left(f1, f2, calcIntersections, next_at, slopes_delta, jmp)
                else:
                    return []
            else:
                # print("next iter")
                more_intersections = find_intersections_left(f1, f2, calcIntersections, next_at, slopes_delta, jmp)
                # print(more_intersections)
                return more_intersections + intersections

        def find_intersections_right(f1, f2, calcIntersections, at, slopes_delta, jmp):
            slopes = [0, 0, 0]
            next_at = at + jmp
            intersections = calcIntersections(f1, f2, at, next_at, 0.001)
            if (len(intersections) == 0):
                for i in range(0, len(slopes_delta)):
                    delta = slopes_delta[i]
                    slopes[i] = calcSlope(lambda x: f1(x) - f2(x), at, delta)
                if (is_more_intersections(slopes)):
                    return find_intersections_right(f1, f2, calcIntersections, next_at, slopes_delta, jmp)
                else:
                    return []
            else:
                # print("next iter")
                more_intersections = find_intersections_right(f1, f2, calcIntersections, next_at, slopes_delta, jmp)
                # print(more_intersections)
                return more_intersections + intersections

        def get_intersections(s, calc_slopes):
            slopes_delta = [0.01, 100, 1000]
            jmp = 100
            left_intersections = find_intersections_left(f1, f2, calc_slopes, s, slopes_delta, jmp)
            right_intersections = find_intersections_right(f1, f2, calc_slopes, s, slopes_delta, jmp)
            return left_intersections + right_intersections

        def clac_aera():
            #calc_slopes = lambda a, b, c, d, e: assignment2.Assignment2().intersections(a, b, c, d, e)
            intersections = assignment2.Assignment2().intersections(f1, f2, 1, 100)

            intersections_len = len(intersections)
            if(intersections_len < 2):
                return np.nan

            area = np.float32(0)
            for i in range(0, intersections_len - 1):
                new_area = abs(self.integrate(f, intersections[i], intersections[i + 1], 100))
                area = area + new_area

            return area

        return clac_aera()


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm

import scipy.integrate as integrate
import matplotlib.pyplot as plt
import tfunctions

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
        self.tfunc_integrate("f(x) = 5", tfunctions.f1, -10, 10, 10)

    def test_integrate_2(self):
        function = np.poly1d([1, 0, 0, -2, 0, 5])
        self.tfunc_integrate("x^5 - 2x^2 + 5", function, 1, 10, 100)

    def test_integrate_3(self):
        self.tfunc_integrate("sin(x^2)", tfunctions.f3, -10, 10, 350)

    def test_integrate_4(self):
        function = np.poly1d([1, 0, 0])
        self.tfunc_integrate("x^2", function, -100, 1000, 10)

    def test_integrate_5(self):
        self.tfunc_integrate("ln(np.ln(x))", tfunctions.f9, 2, 100, 100)

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

        ass3 = Assignment3()
        r = ass3.integrate(f1, s, to, n)
        # print(integrate.simps(f1(points), points))
        expect_area = integrate.quad(lambda x: f1(x), s, to)
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
