"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.

        This function may not work correctly if there is infinite number of
        intersection points.


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
        self.f1 = f1
        self.f2 = f2
        self.a = a
        self.b = b
        self.maxerr = maxerr

        self.jmp_step = 0.1

        self.current_point_x = a
        self.current_point_y = self.f(a)

    def f(self, x: float) -> float:
        return self.f1(x) - self.f2(x)

    def bisection(self, p1: float, p2: float) -> float:
        prev_middle_point = np.nan
        while True:

            middle_point = (p1 + p2) / 2

            if (prev_middle_point == middle_point):
                return np.nan
            else:
                prev_middle_point = middle_point

            f_middle_point = self.f(middle_point)
            if (abs(f_middle_point) < self.maxerr):
                #   print(f"bisection return {middle_point},{f_middle_point}")
                return middle_point

            if (self.f(p1) * f_middle_point < 0):
                p2 = middle_point
            elif (self.f(p2) * f_middle_point < 0):
                p1 = middle_point
            elif (f_middle_point == 0):
                return middle_point
            else:
                return np.nan

    def inc_current_point_next_iter(self):

        while (True):
            self.current_point_x = self.current_point_x + self.jmp_step
            if (self.current_point_x > self.b):
                return
            self.current_point_y = self.f(self.current_point_x)
            if (np.isneginf(self.current_point_y) or np.isinf(self.current_point_y) or np.isnan(
                    self.current_point_y)):
                continue
            if (abs(self.current_point_y) > self.maxerr):
                break

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            # print(f"current_p_x {self.current_point_x} current_p_x {self.current_point_y}")
            if (self.current_point_x > self.b):
                raise StopIteration

            self.current_point_y = self.f(self.current_point_x)
            if (np.isneginf(self.current_point_y) or np.isinf(self.current_point_y) or np.isnan(
                    self.current_point_y)):
                self.current_point_x = self.current_point_x + self.jmp_step
                continue

            if (abs(self.current_point_y) < self.maxerr):
                x = self.current_point_x
                self.inc_current_point_next_iter()
                return x

            next_p_x = self.current_point_x + self.jmp_step
            next_p_y = self.f(next_p_x)

            if (next_p_y * self.current_point_y < 0):
                intersecion = self.bisection(self.current_point_x, next_p_x)

                if (not np.isnan(intersecion)):
                    self.inc_current_point_next_iter()
                    return intersecion

            self.current_point_x = next_p_x


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))


if __name__ == "__main__":
    unittest.main()


