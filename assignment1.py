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

        def interpolated_func_data(points, control_points):
            def interpolated_function(x):
                current_point_x = points[0][0]
                print(current_point_x)
                index = 0
                while(x < current_point_x and index < len(points)-1):
                    current_point_x = points[i][0]
                    index = index + 1
                print(index)
                p0 = points[index]
                p1 = control_points[index]
                p2 = points[index+1]

                t = (x - p0[0]) / (p2[0] - p0[0])
                return cubic_bezier(p0[1], p1[1], p2[1], t)

            return interpolated_function
                
        
        jmp = (b - a) / (n-2)
        #points.append((a + a + jmp) / 2)
        points = [[a, f(a)]] + [[a + jmp*i, f(a + jmp*i)] for i in range(1, n-1)]
       
        print(points)
        cp_x = find_bezier_cubic_control_point((a + a + jmp) / 2, points[0][0], points[1][0])
        cp_y = find_bezier_cubic_control_point(f((a + a + jmp) / 2), points[0][1], points[1][1])
        control_points = [[cp_x, cp_y]]
 
        for i in range(1, n-2):
            cp_x = 2 * points[i][0] - cp_x
            cp_y = 2 * points[i][1] - cp_y
            control_points.append([cp_x, cp_y])
        print(f"control points {control_points}")

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

        d = 300
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 300 + 1)

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

    def t_1(self):
        assignment1 = Assignment1()
        f1 = lambda x: x ** 2
        interpolated = assignment1.interpolate(f1, -1, 1, 4)
        self.assertEqual(1.0, interpolated(1))

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


