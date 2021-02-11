import numpy as np
import time
import random
from functionUtils import AbstractShape


class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape
    def __init__(self):
        pass


class Assignment4:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def area(self, contour: callable, maxerr=0.001) -> np.float32:
        """
        Compute the area of the shape with the given contour.

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """
        def composite_trapezodial(a , b, points_array_y):
            n = points_array_y.size
            h = (b - a) / n
            if(n == 2):
                h = b - a
            sum1_array = [points_array_y[0], points_array_y[-1]]
            sum2_array = []

            for i in range(1, n-1):
                sum2_array.append(points_array_y[i])

            sum1_array.sort()
            sum2_array.sort()
            sum1 = np.sum(sum1_array)
            sum2 = 2 * np.sum(sum2_array)

            return (h * (sum1 + sum2) / 2.0)


        sample_size = 20000
        sample_data = contour(sample_size)
        area = 0
        jmp = int(sample_size / 100)
        for i in range(0, sample_size, jmp):
            a = 0
            if(i != 0):
                a = sample_data[i-1, 0]
            else:
                a = sample_data[i, 0]
            b = sample_data[i+jmp-1, 0]
            if(a > b):
                area = area + -1 * composite_trapezodial(b, a, np.flip(sample_data[i: i+jmp-1][:,1]))
            else:
                area = area + composite_trapezodial(a, b, sample_data[i: i + jmp - 1][:, 1])

        return np.float32(abs(area))

    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape.

        Parameters
        ----------
        sample : callable.
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        An object extending AbstractShape.
        """

        # replace these lines with your solution
        result = MyShape()
        x, y = sample()

        return result


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_delay(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        def sample():
            time.sleep(7)
            return circ()

        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_circle_area_from_contour(self):
        circ = Circle(cx=1, cy=1, radius=1, noise=0.0)
        ass4 = Assignment4()
        T = time.time()
        a_computed = ass4.area(contour=circ.contour, maxerr=0.1)
        T = time.time() - T
        a_true = circ.area()
        self.assertLess(abs((a_true - a_computed) / a_true), 0.1)


if __name__ == "__main__":
    unittest.main()



