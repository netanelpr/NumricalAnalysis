"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random

import matplotlib.pyplot as plt

class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """
        self.derivative_step = 0.01

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> callable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """

        iterator = IntersectionIterable(f1, f2, a, b, maxerr)
        intersetion_list = []
        for intersetion in iterator:
           intersetion_list.append(intersetion)

        return intersetion_list

class IntersectionIterable:

    def __init__(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001):
        self.derivative_step = 0.01
        self.f1 = f1
        self.f2 = f2
        self.a = a
        self.b = b
        self.maxerr = maxerr

        self.init_learning_rate = 0.01
        self.slope_step = 0.01
        self.jmp_step = 0.01

        self.current_point_x = a
        self.current_point_y = self.f(a)
        self.current_point_gradient = self.get_gradient(self.current_point_x, self.current_point_y)

        self.found_last_point = False
        self.cont = 1

    def f(self, x: float) -> float:
        return self.f1(x) - self.f2(x)

    def get_gradient(self, x: float, y: float) -> float:
        gradient = (self.f(x+self.slope_step) - y) / self.slope_step
        if(np.isnan(gradient)): gradient = 0.01
        return gradient

    def set_point_for_next_iter(self, current_root_x: float) -> float:


        """self.current_point_x = current_root_x + self.jmp_step
        self.current_point_y = self.f(self.current_point_x)
        
        self.current_point_gradient = self.get_gradient(self.current_point_x, self.current_point_y)"""

        self.current_point_x = current_root_x
        self.current_point_y = self.f(self.current_point_x)
        while(abs(self.current_point_y) < self.maxerr):
            #print(f"current_p_x {self.current_point_x} current_p_y {self.current_point_y}")

            if(self.current_point_x > self.b):
                break
            self.current_point_x = self.current_point_x + self.jmp_step
            self.current_point_y = self.f(self.current_point_x)
        
        self.current_point_gradient = self.get_gradient(self.current_point_x, self.current_point_y)
        return current_root_x

    def bisection(self, p1: float, p2: float) -> float:
        while True:

            #print(f"bisection p1 {p1} p2 {p2}")
            middle_point = (p1 + p2) / 2
            f_middle_point = self.f(middle_point)

            if(abs(f_middle_point) < self.maxerr):
             #   print(f"bisection return {middle_point},{f_middle_point}")
                return middle_point

            if(self.f(p1) * f_middle_point < 0):
               p2 = middle_point
            elif(self.f(p2) * f_middle_point < 0):
               p1 = middle_point

            

    def __iter__(self):
        return self

   
    def __next__(self):
        self.cont = 1
        learning_rate = self.init_learning_rate

        if(self.found_last_point):
            raise StopIteration

        if(self.current_point_gradient < 0):
            self.cont = -1

        while True:

            """if(np.isneginf(self.current_point_y) or np.isinf(self.current_point_y)):
                self.current_point_x = self.current_point_x + self.jmp_step
                self.current_point_y = self.f(self.current_point_x)
                self.current_point_gradient = self.get_gradient(self.current_point_x, self.current_point_y)"""
           
            next_point_x = self.current_point_x + self.cont * learning_rate * self.current_point_gradient
            next_point_y = self.f(next_point_x)

            if(next_point_x > self.b):
                if(self.current_point_y * self.f(self.b) < 0):
                     ret_point = self.bisection(self.current_point_x, self.b)
                     self.found_last_point = True
                     return ret_point
                raise StopIteration

            if(abs(self.current_point_y) < self.maxerr):
                return self.set_point_for_next_iter(self.current_point_x)

            #print(f"current_p_x {self.current_point_x} current_p_x {self.current_point_y} cont {self.cont} lr {learning_rate}")
            #print(f"next_p_x {next_point_x} next_p_y {next_point_y}")

            if(self.current_point_y * next_point_y < 0):
                return self.set_point_for_next_iter(self.bisection(self.current_point_x, next_point_x))

            next_point_gradient = self.get_gradient(next_point_x, next_point_y)
            #print(f"current_point_gradient {self.current_point_gradient} next_point_gradient {next_point_gradient}")

            if(abs(self.current_point_gradient * next_point_gradient) < 0.001):
                if(self.current_point_gradient < 0):
                    self.cont = -1
                else:
                    self.cont = 1

                self.current_point_x = self.current_point_x + self.jmp_step
                self.current_point_y = self.f(self.current_point_x)
                self.current_point_gradient = self.get_gradient(self.current_point_x, self.current_point_y)
                continue

            if(self.current_point_gradient * next_point_gradient < 0):
                learning_rate = learning_rate / 2
            
            self.current_point_x = next_point_x
            self.current_point_y = next_point_y
            self.current_point_gradient = self.get_gradient(self.current_point_x, self.current_point_y)
            

##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm

import matplotlib.pyplot as plt
import tfunctions

class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        f1 = np.poly1d([1, 0, 0])
        f2 = np.poly1d([2, 0, 0])
        self.tfunc("sqr",f1, f2, -1, 1, 0.001)

    def test_poly(self):
    
        f1, f2 = randomIntersectingPolynomials(10)
        self.tfunc("poly", f1, f2, -1, 1, 0.001)
    
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

        self.tfunc("t1", tfunctions.f1, f2, -10, 10, 0.001, 0)

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

        #self.tfunc("t11", tfunctions.f11, f2, -1, -0.001, 0.001, 2)
        self.tfunc("t11", tfunctions.f11, f2, 0.01, 1, 0.001, 1)

    def test_sqrt(self):

        f_poly = np.poly1d([1, 0, 0])
        f1 = lambda x: np.sqrt(f_poly(x))
        f2 = lambda x: 0

        self.tfunc("sqrt", f1, f2, -100, 100, 0.001, 1)

    def test_12(self):

        function = np.poly1d([1, -6, 8, 0])
        
        f2 = lambda x: 0

        self.tfunc("t10", lambda x: function(x) * np.sin(x), f2, -4, 5, 0.001, 5)

    def tfunc(self, func_name: str, f1: callable, f2: callable, s: float, to: float, maxerr: float, number_of_points=-1, draw=False):

        print(func_name)
        if(draw):
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

        if(number_of_points > -1):
            self.assertEqual(number_of_points, index-1)
        print(f"found {index-1} points")

if __name__ == "__main__":
    unittest.main()
