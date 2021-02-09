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
            def bezier(x):
                current_point_x = points[0][0]
                #print(current_point_x)
                index = 0
                while(x > current_point_x and index < len(points)-1):
                    #print(f"{x} {current_point_x}")
                    index = index + 1
                    current_point_x = points[index][0]

                index = index - 1
                #print(len(points))
                #print(len(control_points))
                #print(f"index {index}")
                p0 = points[index]
                p1 = control_points[index]
                p2 = points[index+1]

                t = (x - p0[0]) / (p2[0] - p0[0])
                #print((x - p0[0]))
                #print(p2[0] - p0[0])
                #print(f"p0 {p0} p1 {p1} p2 {p2}")
                #print(f"t {t} x {x}")
                return cubic_bezier(p0[1], p1[1], p2[1], t)

            def lagrange3(x):
                def li_i_2(x, p1_x, p2_x, p3_x):
                    return (x - p2_x) / (p1_x - p2_x) * (x - p3_x) / (p1_x - p3_x)

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

                c_points = [points[index], control_points[index], points[index+1]]
                #print(c_points)
                y_x = 0
                for i in range(3):
                    p1 = c_points[i-3]
                    p2 = c_points[i-2]
                    p3 = c_points[i-1]
                    y_x = y_x + p1[1] * li_i_2(x, p1[0], p2[0], p3[0])

                return y_x

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

            return lagrange2
        
        

        def one_control_point():
            jmp = (b - a) / (n-2)
            points = [[a, f(a)]] + [[a + jmp*i, f(a + jmp*i)] for i in range(1, n-1)]
       
            #print(points)
            #print(len(points))
            cp_x = find_bezier_cubic_control_point((a + a + jmp) / 2, points[0][0], points[1][0])
            cp_y = find_bezier_cubic_control_point(f((a + a + jmp) / 2), points[0][1], points[1][1])
            control_points = [[cp_x, cp_y]]
 
            for i in range(1, n-2):
                cp_x = 2 * points[i][0] - cp_x
                cp_y = 2 * points[i][1] - cp_y
                control_points.append([cp_x, cp_y])
            #print(f"\n\ncontrol points {control_points} len {len(control_points)}\n\n")
            return (points, control_points)

        def half_control_point():
            number_of_points = int((n/2)) - 1
            jmp = (b - a) / number_of_points
            points = [[a, f(a)]]
            control_points = []
            point_x = a + jmp
            for i in range(1, number_of_points+1):
                points.append([point_x, f(point_x)])
                point_x = point_x + jmp

            for i in range(0, number_of_points):
                p1_x = points[i][0]
                p1_y = points[i][1]
                p2_x = points[i+1][0]
                p2_y = points[i+1][1]
                cp_x = find_bezier_cubic_control_point((p2_x + p1_x) / 2, p1_x, p2_x)
                cp_y = find_bezier_cubic_control_point(f((p2_y + p1_y) / 2), p1_y, p2_y)
                control_points.append([cp_x, cp_y]) 

            #print(f"points {points} len {len(points)}")
            #print(f"control points {control_points} len {len(control_points)}")
            return (points, control_points)
 
        def all_points():
            number_of_points = n
            jmp = (b - a) / number_of_points
            points = []
            point_x = a
            for i in range(0, number_of_points):
                points.append([point_x, f(point_x)])
                point_x = point_x + jmp

            #print(f"points {points} len {len(points)}")
            #print(f"control points {control_points} len {len(control_points)}")
            return (points, [])

        points, control_points = all_points()
        return interpolated_func_data(points, control_points)

##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm
import tfunctions

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
            xs1 = []
            if(-10 < 0):
                for number in np.random.random(200):
                    if (number < 0.5):
                        xs1.append(-1)
                    else:
                        xs1.append(1)
            else:
                xs1 = 1
            xs = (np.random.random(200))* xs1 * ((-10 - 10) / 2)
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

    def tfunc(self, function_name, f, s, to, number_of_dots):
        assignment1 = Assignment1()

        interpolated = assignment1.interpolate(f, s, to, number_of_dots)

        xs1 = []
        if(s < 0):
            for number in np.random.random(20):
                if (number < 0.5):
                    xs1.append(-1)
                else:
                    xs1.append(1)
        else:
            xs1 = 1
        xs = (np.random.random(20))* xs1 * ((to - s) / 2)
        r_err = 0
        for x in xs:
            y2 = interpolated(x)
            y = f(x)
            #print(x)
            #print(f"{y} {y2}")
            r_err = r_err + abs((y - y2) / y)
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


