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
        return iterator

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
        self.jmp_step = 0.1

        self.current_point_x = a
        self.current_point_y = self.f(a)
        self.current_point_gradient = self.get_gradient(self.current_point_x, self.current_point_y)

        self.found_last_point = False
        self.cont = 1

    def f(self, x: float) -> float:
        return self.f1(x) - self.f2(x)

    def get_gradient(self, x: float, y: float) -> float:
        return (self.f(x+self.slope_step) - y) / self.slope_step

    def set_point_for_next_iter(self, current_root_x: float) -> float:


        """self.current_point_x = current_root_x + self.jmp_step
        self.current_point_y = self.f(self.current_point_x)
        
        self.current_point_gradient = self.get_gradient(self.current_point_x, self.current_point_y)"""

        self.current_point_x = current_root_x
        self.current_point_y = self.f(self.current_point_x)
        while(abs(self.current_point_y) < self.maxerr):
            #print(f"current_p_x {self.current_point_x} current_p_x {self.current_point_y}")
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

            if(abs(self.current_point_y) < self.maxerr):
                return self.set_current_point_ret_given_point(self.current_point_x)

            next_point_x = self.current_point_x + self.cont * learning_rate * self.current_point_gradient
            next_point_y = self.f(next_point_x)

            if(next_point_x == self.current_point_x):
                raise Exception
            if(next_point_x > self.b):
                if(self.current_point_y * self.f(self.b) < 0):
                     ret_point = self.bisection(self.current_point_x, self.b)
                     self.found_last_point = True
                     return ret_point
                raise StopIteration

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

             #   print(f"cont current_p_x {self.current_point_x} current_p_x {self.current_point_y}")
                self.current_point_x = self.current_point_x + 0.1
                self.current_point_y = self.f(self.current_point_x)
                self.current_point_gradient = self.get_gradient(self.current_point_x, self.current_point_y)
              #   print(f"current_p_x {self.current_point_x} current_p_x {self.current_point_y}")
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

class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        """p_x = np.arange(-1.5, 1.5, 0.1)
        p_y = f1(p_x) - f2(p_x)
        plt.plot(p_x, p_y)
        plt.show()"""
        print("sqr")
        X = ass2.intersections(f1, f2, -1.5, 1.5, maxerr=0.001)
        index = 1 
        for x in X:
            print(f"{index} {x}")
            index = index + 1
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
        print(index)

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)
    
        """p_x = np.arange(-1, 1, 0.1)
        p_y = f1(p_x) - f2(p_x)
        plt.plot(p_x, p_y)
        plt.show()"""

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
        print("p")
        index = 1 
        for x in X:
            print(f"{index} {x}")
            index = index + 1
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
        print(index)

if __name__ == "__main__":
    unittest.main()
