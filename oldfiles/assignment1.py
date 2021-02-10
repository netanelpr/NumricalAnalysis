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
            #bezier_function = lambda p0, p1, p2, t: p0 * (1 - t) ** 2 + 2 * p1 * t * (1 - t) + p2 * t ** 2
            #return [bezier_function(p0[0], p1[0], p2[0], t), bezier_function(p0[1], p1[1], p2[1], t)]
            return p0 * (1 - t) ** 2 + 2 * p1 * t * (1 - t) + p2 * t ** 2

        def interpolated_func_data(points, control_points, use_lagrange=False):
            def lagrange2(x):
                def li_i_1(x, p1_x, p2_x):
                    return (x - p2_x) / (p1_x - p2_x)

                current_point_x = points[0][0]
                #print(current_point_x)
                index = 0
                while(x > current_point_x and index < len(points)-1):
                    #print(f"{x} {current_point_x}")
                    index = index + 1
                    current_point_x = points[index][0]

                index = index - 1
                #print(x)
                #print(len(points))
                #print(len(control_points))
                #print(f"index {index}")
                c_points = [points[index], points[index+1]]
                #print(c_points)
                y_x = 0
                for i in range(2):
                    p1 = c_points[i-2]
                    p2 = c_points[i-1]
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
                #print(current_point_x)
                index = 0
                while(x > current_point_x and index < len(points)-1):
                    #print(f"{x} {current_point_x}")
                    index = index + 1
                    current_point_x = points[index][0]

                index = index - 1
                #print(x)
                #print(len(points))
                #print(len(control_points))
                #print(f"index {index}")
                c_points = [points[index], points[index+1]]
                slope = control_points[index]
                #print(f"{c_points} {slope}")
                y_x = 0
                for i in range(2):
                    p1 = c_points[i-2]
                    p2 = c_points[i-1]
                    y_x = y_x + p1[1] * h_i(x, p1[0], p2[0]) + slope * h2_i(x, p1[0], p2[0])

                return y_x

            if(not use_lagrange):
                return hermite
            return lagrange2
        
        def half_derivative():
            number_of_points = int((n/2)) - 1
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

            #print(f"points {points} len {len(points)}")
            #print(f"control points {control_points} len {len(control_points)}")
            return (points, control_points)
 
        def all_points():
            number_of_points = n - 1
            jmp = (b - a) / number_of_points
            points = []
            point_x = a
            for i in range(0, number_of_points + 1):
                points.append([point_x, f(point_x)])
                point_x = point_x + jmp

            #print(f"points {points} len {len(points)}")
            #print(f"control points {control_points} len {len(control_points)}")
            return (points, [])


        if(n < 5):
            points, control_points = all_points()
            return interpolated_func_data(points, control_points, True)
        else:
            points, control_points = half_derivative()
            return interpolated_func_data(points, control_points)

##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm
import tfunctions
import matplotlib.pyplot as plt

class TestAssignment1(unittest.TestCase):

    """def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 300
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 300 + 1)
            xs = np.random.random(200)
            err = 0
            for x in xs:
                #print(x)
                yy = ff(x)
                y = f(x)
                err += abs((y - yy) / y)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)"""

    def test_1(self):
        self.tfunc("sqr", lambda x: x ** 2, -5, 5, 10)

    def test_2(self):
        self.tfunc("t2", lambda x: tfunctions.f2(x), -10, 10, 10)

    def test_3(self):
        self.tfunc("t3", lambda x: tfunctions.f3(x), -20, 20, 200)

    def test_7(self):
        self.tfunc("t7", lambda x: tfunctions.f7(x), 1.01, 5, 200)

    def test_12(self):
        self.tfunc("t_2_points", lambda x: x ** 2, -5, 5, 2, True)

    def test_13(self):
        self.tfunc("t_3_points", lambda x: x ** 2, -5, 5, 3, True)

    def test_14(self):
        self.tfunc("t_4_points", lambda x: x ** 2, -5, 5, 4, True)

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

    """def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)"""


if __name__ == "__main__":
    unittest.main()


