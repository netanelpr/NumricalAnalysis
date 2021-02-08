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

from assignment2 import Assignment2

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
            sum1_array = [a]
            sum2_array = []
            sum3_array = [b]

            x = a + h
            for i in range(1, n):
                point = np.float32(f(x))

                if(i % 2 == 0):
                    sum1_array.append(point)
                    sum3_array.append(point)
                else:
                    sum2_array.append(point)

                x = x + h

            sum1_array.sort()
            sum2_array.sort()
            sum3_array.sort()

            sum1 = np.sum(sum1_array, dtype=np.float32)
            sum2 = np.float32(4.0) * np.sum(sum2_array, dtype=np.float32)
            sum3 = np.sum(sum3_array, dtype=np.float32)

            return (h * (sum1 + sum3 + sum2)) / np.float32(3.0)

        return composite_simpson(np.float32(a), np.float32(b))

    def areabetween(self, f1: callable, f2: callable): #-> np.float32:
        """
        Finds the area enclosed between two functions. This method finds 
        all intersection points between the two functions to work correctly. 
        
        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        Note, there is no such thing as negative area. 
        
        In order to find the enclosed area the given functions must intersect 
        in at least two points. If the functions do not intersect or intersect 
        in less than two points this function returns NaN.  
        

        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis

        """

        def calcSlope(f, p, delta):
            y_p = f(p)
            y_to = f(p + delta)
            return (y_to - y_p) / delta

        def is_more_intersections(slopes_array):
            if(slopes_array[0] < 0):
                for i in range(1, len(slopes_array)):
                    if(slopes_array[i-1] < slopes_array[i]):
                        return True
            else:
                for i in range(1, len(slopes_array)):
                    if(slopes_array[i-1] > slopes_array[i]):
                        return True
            return False

        def find_intersections_left(f1, f2, calcIntersections, at, slopes_delta, jmp):
            slopes = [0, 0, 0]
            next_at = at - jmp
            intersections = calcIntersections(f1, f2, next_at, at, 0.001)
            #print(intersections)
            if(len(intersections) == 0):
                for i in range(0, len(slopes_delta)):
                    delta = slopes_delta[i]
                    slopes[i] = calcSlope(lambda x: f1(x) - f2(x), at, (delta * -1)) * -1
                if(is_more_intersections(slopes)):
                    return find_intersections_left(f1, f2, calcIntersections, next_at, slopes_delta, jmp)
                else:
                    return []
            else:
                #print("next iter")
                more_intersections = find_intersections_left(f1, f2, calcIntersections, next_at, slopes_delta, jmp)
                #print(more_intersections)
                return more_intersections + intersections

        def find_intersections_right(f1, f2, calcIntersections, at, slopes_delta, jmp):
            slopes = [0, 0, 0]
            next_at = at + jmp
            intersections = calcIntersections(f1, f2, at, next_at, 0.001)
            #print(intersections)
            if(len(intersections) == 0):
                for i in range(0, len(slopes_delta)):
                    delta = slopes_delta[i]
                    slopes[i] = calcSlope(lambda x: f1(x) - f2(x), at, delta) 
                if(is_more_intersections(slopes)):
                    return find_intersections_right(f1, f2, calcIntersections, next_at, slopes_delta, jmp)
                else:
                    return []
            else:
                #print("next iter")
                more_intersections = find_intersections_right(f1, f2, calcIntersections, next_at, slopes_delta, jmp)
                #print(more_intersections)
                return more_intersections + intersections

        def get_intersections(s, calc_slopes):
            slopes_delta = [0.01, 100, 1000]
            jmp = 100
            left_intersections = find_intersections_left(f1, f2, calc_slopes, s, slopes_delta, jmp)
            right_intersections = find_intersections_right(f1, f2, calc_slopes, s, slopes_delta, jmp)
            return left_intersections + right_intersections

        """def get_intersection_points(self, calcIntersections):
            start = -10
            to = 10
            range_jmp = 100
            f = lambda x: f1(x) - f2(x)
            intersections = calcIntersections(f1, f2, start, to, 0.001)
            slopes_delta = [0.01, 100, 1000]
            left_most_slopes = [0, 0, 0]
            right_most_slopes = [0, 0, 0]
            left_point_found = False
            right_point_found = False
            
            while not (left_point_found and right_point_found):
                new_intersections = calcIntersections(f1, f2, start, to, 0.001)
                
                if(intersections.size() < 2):
                    start = start - range_jmp
                    to = to + range_jmp
                    continue
                
                left_most_inters = intersections[0]
                right_most_inters = intersections[-1]

                for i in range(0, slopes_delta.size()):
                    delta = slopes_delta[i]
                    if(not left_point_found):
                        left_most_slopes[i] = (calcSlope(f, left_most_inters, (delta * -1)) * -1)
                    if(not right_point_found):
                        right_most_slopes[i] = calcSlope(f, right_most_inters, delta)
                 

                if((not left_point_found) and is_more_intersections(left_most_slopes)):
                    start = start - range_jmp
                else:
                    left_point_found = True
                if((not right_point_found) and is_more_intersections(right_most_slopes)):
                    start = start + range_jmp
                else:
                    right_point_found = True"""      

        assignment2 = Assignment2()
        area = []

        #f1 = np.poly1d([-1, 0, 90])
        f1 = np.poly1d([3, 7, 1, 4, 0, -4])
        f2 = lambda x: 0
        calc_slopes = lambda a, b, c, d, e : assignment2.intersections(a, b, c, d, e)
        print(get_intersections(0, calc_slopes))

        return 1


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm

import scipy.integrate as integrate
import matplotlib.pyplot as plt

class TestAssignment3(unittest.TestCase):

    def test_integrate_float32(self):
        f1 = np.poly1d([1, 0, -1])

        self.tfunc("poly", f1, -1, 1, 10, 0.001)

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = strong_oscilations()
        r = ass3.integrate(f1, 0.05, 10, 100)

        self.assertGreaterEqual(0.001, (r - 2.0998 * 10 ** 116) / 2.0998 * 10 ** 116)

    def tfunc(self, func_name: str, f1: callable, s: float, to: float, n: int, maxerr: float, draw=False):

        if(draw):
            p_x = np.arange(s * 1.1, to * 1.1, 0.1)
            p_y = f1(p_x) - f2(p_x)
            plt.plot(p_x, p_y)
            plt.show()
       
        ass3 = Assignment3()
        r = ass3.integrate(f1, s, to, n)
        points = np.arange(s, to, ((to-s)/n))
        #print(integrate.simps(f1(points), points))
        expect_aera = integrate.quad(lambda x: f1(x), s, to)
        self.assertEqual(r.dtype, np.float32)
        self.assertGreaterEqual(maxerr, abs(r - expect_aera[0]), func_name)
    

if __name__ == "__main__":
    #unittest.main()
    a3 = Assignment3()
    f1 = np.poly1d([-1, 0, 90])
    f2 = lambda x: 0
    a3.areabetween(f1, f2)

