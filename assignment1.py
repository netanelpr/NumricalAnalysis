"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random


class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        starting to interpolate arbitrary functions.
        """

        pass

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time.
        The assignment will be tested on variety of different functions with
        large n values.

        Interpolation error will be measured as the average absolute error at
        2*n random points between a and b. See test_with_poly() below.

        Note: It is forbidden to call f more than n times.

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.**

        Note: sometimes you can get very accurate solutions with only few points,
        significantly less than n.

        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """

        def interpolated_func_data(points, control_points, use_lagrange=False):
            def lagrange2(x):
                def li_i_1(x, p0_x, p1_x):
                    return (x - p1_x) / (p0_x - p1_x)

                current_point_x = points[0, 0]
                len = int(points.size / points.ndim)

                index = 0
                t = (x - a) / (b - a)
                index = int(t * (n - 1))
                if(index == len - 1):
                    index = index - 1

                c_points = [points[index], points[index + 1]]
                y_x = 0
                for i in range(2):
                    p1 = c_points[i - 2]
                    p2 = c_points[i - 1]
                    y_x = y_x + p1[1] * li_i_1(x, p1[0], p2[0])

                return y_x

            def hermite(x):
                def li_i_1(x, p0_x, p1_x):
                    return (x - p1_x) / (p0_x - p1_x)

                def li_i_1_d(p0_x, p1_x):
                    return 1 / (p0_x - p1_x)

                def h_i(x, p0_x, p1_x):
                    return (1 - 2 * (x - p0_x) * li_i_1_d(p0_x, p1_x)) * (li_i_1(x, p0_x, p1_x) ** 2)

                def h2_i(x, p0_x, p1_x):
                    return (x - p0_x) * (li_i_1(x, p0_x, p1_x) ** 2)

                current_point_x = points[0, 0]
                index = 0
                len = int(points.size / points.ndim)

                index = 0
                t = (x - a) / (b - a)
                index = int(t * (len - 1))
                if(index == len - 1):
                    index = index - 1

                c_points = [points[index], points[index + 1]]
                slope = control_points[index]
                y_x = 0
                for i in range(2):
                    p1 = c_points[i - 2]
                    p2 = c_points[i - 1]
                    y_x = y_x + p1[1] * h_i(x, p1[0], p2[0]) + slope * h2_i(x, p1[0], p2[0])

                return y_x

            if (not use_lagrange):
                return hermite
            return lagrange2

        def half_derivative():
            number_of_points = int((n / 2)) - 1
            jmp = (b - a) / number_of_points
            points = np.empty([number_of_points + 1, 2])
            control_points = np.empty(number_of_points + 1)
            point_x = a
            derivative_jmp = 0.001
            for i in range(0, number_of_points + 1):
                y_point = f(point_x)
                points[i, 0] = point_x
                points[i, 1] = y_point
                slope = (f(point_x + derivative_jmp) - y_point) / derivative_jmp
                control_points[i] = slope
                point_x = point_x + jmp
            return (points, control_points)

        def all_points():
            number_of_points = n - 1
            jmp = (b - a) / number_of_points
            points = np.empty([number_of_points + 1, 2])
            point_x = a
            for i in range(0, number_of_points + 1):
                points[i, 0] = point_x
                points[i, 1] = f(point_x)
                point_x = point_x + jmp
            return (points, None)

        if (n == 1):
            return lambda x: a
        elif(n < 5):
            points, control_points = all_points()
            return interpolated_func_data(points, control_points, True)
        else:
            points, control_points = half_derivative()
            return interpolated_func_data(points, control_points)


##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)

if __name__ == "__main__":
    unittest.main()
