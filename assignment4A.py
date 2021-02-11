"""
In this assignment you should fit a model function of your choice to data
that you sample from a given function.

The sampled data is very noisy so you should minimize the mean least squares
between the model you fit and the data points you sample.

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You
must make sure that the fitting function returns at most 5 seconds after the
allowed running time elapses. If you take an iterative approach and know that
your iterations may take more than 1-2 seconds break out of any optimization
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools
for solving this assignment.

"""

import numpy as np
import time
import random
import torch
import time


class Assignment4A:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        self.current_time = 0

    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape.

        Parameters
        ----------
        f : callable.
            A function which returns an approximate (noisy) Y value given X.
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        a function:float->float that fits f between a and b
        """

        def get_samples(sample_size: int):
            sample_array_x = np.empty(sample_size + 1)
            sample_array_y = np.empty(sample_size + 1)
            sample_jmps = (b - a) / sample_size
            current_point_x = a
            for i in range(sample_size+1):
                if(self.current_time < maxtime):
                    sample_array_x[i] = current_point_x

                    t = time.time()
                    sample_array_y[i] = f(current_point_x)
                    self.current_time = self.current_time + (time.time() - t)

                    current_point_x = current_point_x + sample_jmps
                else:
                    return (None, None)
            return (sample_array_x, sample_array_y)

        def polynomial_coefficients():
            sample_array_x, sample_array_y = get_samples((10000))

            list_array = []
            if(self.current_time < maxtime):
                t = time.time()
                for i in range(d+1):
                    list_array.append(torch.Tensor(sample_array_x ** i))
                self.current_time = self.current_time + (time.time() - t)

            else:
                return None

            ATranspose = torch.stack(list_array)
            A = ATranspose.T
            Y = torch.stack([torch.Tensor(sample_array_y)]).T

            return (ATranspose.mm(A)).inverse().mm(ATranspose).mm(Y)

        def fitting_function(coefficients):
            def inner_function(x):
                y = coefficients[0]
                for i in range(1, d + 1):
                    y = y + coefficients[i] * (x ** i)
                return y
            return inner_function

        coefficients = polynomial_coefficients()
        return fitting_function(coefficients)


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1, 1, 1))
        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    def test_delay(self):
        f = DELAYED(7)(NOISY(0.01)(poly(1, 1, 1)))

        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1, 1, 1)
        nf = NOISY(1)(f)
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse = 0
        for x in np.linspace(0, 1, 1000):
            self.assertNotEquals(f(x), nf(x))
            mse += (f(x) - ff(x)) ** 2
        mse = mse / 1000
        print(mse)


if __name__ == "__main__":
    unittest.main()
