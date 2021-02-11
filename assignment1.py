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

        def get_points():
            jmp = (b - a) / (n - 1)
            points = np.empty([n, 2])
            point_x = a
            for i in range(0, n):
                points[i, 0] = point_x
                points[i, 1] = f(point_x)
                point_x = point_x + jmp
            return points

        def create_coefficents_matrix():
            coff = 4 * np.identity(n - 1)
            coff[0, 0] = 2
            np.fill_diagonal(coff[1:], 1)
            np.fill_diagonal(coff[:, 1:], 1)
            coff[n - 2, n - 3] = 2
            coff[n - 2, n - 2] = 7

            return coff

        def build_sol_vector(points):
            size = n - 1
            vector_x = np.empty(size)
            vector_y = np.empty(size)

            vector_x[0] = points[0, 0] + 2 * points[1, 0]
            vector_y[0] = points[0, 1] + 2 * points[1, 1]

            vector_x[size - 1] = 8 * points[size - 1, 0] + points[size, 0]
            vector_y[size - 1] = 8 * points[size - 1, 1] + points[size, 1]

            for i in range(1, size - 1):
                vector_x[i] = 4 * points[i, 0] + 2 * points[i + 1, 0]
                vector_y[i] = 4 * points[i, 1] + 2 * points[i + 1, 1]

            return (vector_x, vector_y)

        def solve_matrix(coff_matrix, points):
            b_array = np.copy(np.diag(coff_matrix))
            size = len(b_array) - 1
            a_array = np.copy(np.diag(coff_matrix, k=-1))
            c_array = np.copy(np.diag(coff_matrix, k=1))
            d_array = np.copy(points)
            w = np.float64(0.0)

            for i in range(1, size + 1):
                w = a_array[i - 1] / b_array[i - 1]
                b_array[i] = b_array[i] - w * c_array[i - 1]
                d_array[i] = d_array[i] - w * d_array[i - 1]

            x_array = np.empty(size + 1, dtype=np.float64)
            x_array[size] = d_array[size] / b_array[size]
            for i in range(size - 1, -1, -1):
                x_array[i] = (d_array[i] - c_array[i] * x_array[i + 1]) / b_array[i]

            return x_array

        def get_control_points(points):
            coefficents_matrix = create_coefficents_matrix()
            vector_x, vector_y = build_sol_vector(points)
            coff_from_x = solve_matrix(coefficents_matrix, vector_x)
            coff_from_y = solve_matrix(coefficents_matrix, vector_y)

            A = np.stack((coff_from_x, coff_from_y), axis=1)
            B = np.empty([n - 1, 2])
            for i in range(n - 2):
                B[i] = 2 * points[i + 1] - A[i + 1]
            B[n - 2] = (A[n - 2] + points[n - 1]) / 2

            return A, B

        def beizer_3_curve(p0, p1, p2, p3, t):
            return p0 * np.power(1 - t, 3) + 3 * p1 * np.power(1 - t, 2) * t + 3 * p2 * (1 - t) * np.power(t,
                                                                                                           2) + p3 * np.power(
                t, 3)

        def bezier(points, A, B):
            def inner(x):
                t = (x - a) / (b - a)
                index = int(t * (n - 1))

                if(index == int(points.size / points.ndim) - 1):
                    index = index - 1

                t = (x - points[index, 0]) / (points[index + 1, 0] - points[index, 0])
                return beizer_3_curve(points[index, 1], A[index][1], B[index][1], points[index + 1, 1], t)

            return inner

        points = get_points()
        A, B = get_control_points(points)
        return bezier(points, A, B)

##########################################################################

import unittest
from functionUtils import *
from tqdm import tqdm
import tfunctions
import matplotlib.pyplot as plt

class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 50)

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

    def test_1(self):
        self.tfunc("sqr", lambda x: x ** 2, -5, 5, 10)

    def test_2(self):
        self.tfunc("t2", lambda x: tfunctions.f2(x), -10, 10, 10)

    def test_3(self):
        self.tfunc("t3", lambda x: tfunctions.f3(x), -20, 20, 200)

    def test_7(self):
        self.tfunc("t7", lambda x: tfunctions.f7(x), 1.01, 5, 200)

    def test_12(self):
        self.tfunc("t_2_points", lambda x: x ** 2, -5, 5, 2)

    def test_13(self):
        self.tfunc("t_3_points", lambda x: x ** 2, -5, 5, 3)

    def test_14(self):
        self.tfunc("t_4_points", lambda x: x ** 2, -5, 5, 4)

    def tfunc(self, function_name, f, s, to, number_of_dots, draw=False):
        assignment1 = Assignment1()
        interpolated = assignment1.interpolate(f, s, to, number_of_dots)

        print(function_name)

        xs = (np.random.random(20)) * ((to - s)) - (to - s) / 2
        xs = np.concatenate((np.array([s, to]), xs))
        ys = []
        y2s = []
        r_err = 0
        for x in xs:
            if(x < s):
                x = s + 1

            y2 = interpolated(x)
            y2s.append(y2)
            y = f(x)
            ys.append(y)
            #print(x)
            #print(f"{x} {y} {y2} {abs((y - y2) / y)}")
            r_err = r_err + abs((y - y2) / y)

        if(draw):
            #x2s = [number for number in np.linspace(s * 1.0, to * 1.0)]
            #y2s = [f(number) for number in x2s]
            plt.plot(xs, y2s, "bo")
            plt.plot(xs, ys, "ro")
            plt.show()

        print(r_err / 20)

if __name__ == "__main__":
    unittest.main()
