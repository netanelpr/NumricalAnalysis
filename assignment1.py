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

        def find_bezier_cubic_control_point(number, p0, p2):
            return (number - (p0 * 0.25 + p2 * 0.25)) * 2

        def cubic_bezier(p0, p1, p2, t):
            # bezier_function = lambda p0, p1, p2, t: p0 * (1 - t) ** 2 + 2 * p1 * t * (1 - t) + p2 * t ** 2
            # return [bezier_function(p0[0], p1[0], p2[0], t), bezier_function(p0[1], p1[1], p2[1], t)]
            return p0 * (1 - t) ** 2 + 2 * p1 * t * (1 - t) + p2 * t ** 2

        def interpolated_func_data(points, control_points, use_lagrange=False):
            def lagrange2(x):
                def li_i_1(x, p1_x, p2_x):
                    return (x - p2_x) / (p1_x - p2_x)

                current_point_x = points[0][0]
                # print(current_point_x)
                index = 0
                while (x > current_point_x and index < len(points) - 1):
                    # print(f"{x} {current_point_x}")
                    index = index + 1
                    current_point_x = points[index][0]

                index = index - 1
                # print(x)
                # print(len(points))
                # print(len(control_points))
                # print(f"index {index}")
                c_points = [points[index], points[index + 1]]
                # print(c_points)
                y_x = 0
                for i in range(2):
                    p1 = c_points[i - 2]
                    p2 = c_points[i - 1]
                    y_x = y_x + p1[1] * li_i_1(x, p1[0], p2[0])

                return y_x

            def hermite(x):
                def li_i_1(x, p1_x, p2_x):
                    return (x - p2_x) / (p1_x - p2_x)

                def li_i_1_d(p1_x, p2_x):
                    return 1 / (p1_x - p2_x)

                def h_i(x, p1_x, p2_x):
                    return (1 - (x - p1_x) * li_i_1_d(p1_x, p2_x)) * (li_i_1(x, p1_x, p2_x) ** 2)

                def h2_i(x, p1_x, p2_x):
                    return (x - p1_x) * (li_i_1(x, p1_x, p2_x) ** 2)

                current_point_x = points[0][0]
                # print(current_point_x)
                index = 0
                while (x > current_point_x and index < len(points) - 1):
                    # print(f"{x} {current_point_x}")
                    index = index + 1
                    current_point_x = points[index][0]

                index = index - 1
                # print(x)
                # print(len(points))
                # print(len(control_points))
                # print(f"index {index}")
                c_points = [points[index], points[index + 1]]
                slope = control_points[index]
                # print(f"{c_points} {slope}")
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
            points = []
            control_points = []
            point_x = a
            derivative_jmp = 0.001
            for i in range(0, number_of_points + 1):
                y_point = f(point_x)
                points.append([point_x, y_point])
                slope = (f(point_x + derivative_jmp) - y_point) / derivative_jmp
                control_points.append(slope)
                point_x = point_x + jmp

                # print(f"points {points} len {len(points)}")
            # print(f"control points {control_points} len {len(control_points)}")
            return (points, control_points)

        def all_points():
            number_of_points = n - 1
            jmp = (b - a) / number_of_points
            points = []
            point_x = a
            for i in range(0, number_of_points + 1):
                points.append([point_x, f(point_x)])
                point_x = point_x + jmp

            # print(f"points {points} len {len(points)}")
            # print(f"control points {control_points} len {len(control_points)}")
            return (points, [])

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
                t = (x - points[index, 0]) / (points[index + 1, 0] - points[index, 0])
                return beizer_3_curve(points[index, 1], A[index][1], B[index][1], points[index + 1, 1], t)

            return inner

        points = get_points()
        A, B = get_control_points(points)
        return bezier(points, A, B)

def interpolate(f: callable, a: float, b: float, n: int) -> callable:
    def get_points():
        jmp = (b - a) / (n-1)
        points = np.empty([n, 2])
        point_x = a
        for i in range(0,  n):
            points[i, 0] = point_x
            points[i, 1] = f(point_x)
            point_x = point_x + jmp
        return points

    def create_coefficents_matrix():
        coff = 4 * np.identity(n-1)
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

        vector_x[size-1] = 8 * points[size-1, 0] + points[size, 0]
        vector_y[size-1] = 8 * points[size-1, 1] + points[size, 1]

        for i in range(1, size-1):
            vector_x[i] = 4 * points[i, 0] + 2 * points[i+1, 0]
            vector_y[i] = 4 * points[i, 1] + 2 * points[i+1, 1]

        return (vector_x, vector_y)

    def solve_matrix(coff_matrix, points):
        b_array = np.copy(np.diag(coff_matrix))
        size = len(b_array) - 1
        a_array = np.copy(np.diag(coff_matrix, k=-1))
        c_array = np.copy(np.diag(coff_matrix, k=1))
        d_array = np.copy(points)
        w = np.float64(0.0)

        for i in range(1, size+1):
            w = a_array[i-1] / b_array[i - 1]
            b_array[i] = b_array[i] - w * c_array[i - 1]
            d_array[i] = d_array[i] - w * d_array[i - 1]

        x_array = np.empty(size+1, dtype=np.float64)
        x_array[size] = d_array[size] / b_array[size]
        for i in range(size-1, -1, -1):
            x_array[i] = (d_array[i] - c_array[i] * x_array[i + 1]) / b_array[i]

        return x_array

    def get_control_points(points):
        coefficents_matrix = create_coefficents_matrix()
        vector_x, vector_y = build_sol_vector(points)
        coff_from_x = solve_matrix(coefficents_matrix, vector_x)
        coff_from_y = solve_matrix(coefficents_matrix, vector_y)

        A = np.stack((coff_from_x, coff_from_y), axis=1)
        B = np.empty([n-1, 2])
        for i in range(n - 2):
            B[i] = 2 * points[i + 1] - A[i + 1]
        B[n - 2] = (A[n - 2] + points[n-1]) / 2

        return A, B

    def beizer_3_curve(p0, p1, p2, p3, t):
        return p0 * np.power(1 - t, 3) + 3 * p1 * np.power(1 - t, 2) * t + 3 * p2 * (1 - t) * np.power(t, 2) + p3 * np.power(t, 3)

    def bezier(points, A, B):
        def inner(x):
            t = (x - a) / (b - a)
            index = int(t * (n - 1))
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
