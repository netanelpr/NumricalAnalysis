"""
In this assignment you should fit a model function of your choice to data 
that you sample from a contour of given shape. Then you should calculate
the area of that shape. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you know that your iterations may take more 
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment. 
Note: !!!Despite previous note, using reflection to check for the parameters 
of the sampled function is considered cheating!!! You are only allowed to 
get (x,y) points from the given shape by calling sample(). 
"""

import numpy as np
import time
import random
from functionUtils import AbstractShape
import assignment3

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

    def area(self, contour: callable, maxerr=0.001)->np.float32:
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

        def get_order_sample_and_directions(sample_data, sample_size):
            direction = 1
            prev_point_x = sample_data[1][0]
            if (sample_data[0][0] > prev_point_x):
                direction = -1

            indices = []
            direction_list = []
            for i in range(2, sample_size):
                current_point_x = sample_data[i][0]
                if (direction == 1):
                    if (current_point_x < prev_point_x):
                        indices.append(i)
                        direction_list.append(direction)
                        direction = direction * -1
                elif (current_point_x > prev_point_x):
                    indices.append(i)
                    direction_list.append(direction)
                    direction = direction * -1
                prev_point_x = current_point_x
            direction_list.append(direction)
            sample_data = np.split(sample_data, indices)
            return (sample_data, direction_list)



        intergral = assignment3.Assignment3()
        sample_size = 10000
        sample_data = contour(sample_size)
        sample_data, direction_list = get_order_sample_and_directions(sample_data, sample_size)

        area = 0
        for i in range(len(direction_list)):
            data_array = np.sort(sample_data[i], axis=0)
            data_array_x = np.array(data_array[:,0])
            data_array_y = np.array(data_array[:,1])
            function = np.poly1d(np.polyfit(data_array_x, data_array_y, 2))
            area = area + direction_list[i] * -1 * intergral.integrate(function, data_array_x[0], data_array_x[-1], 100)

            """size_to_split = int(data_array_x.size / 10)
            index = 0

            while(index < data_array_x.size):
                size = 10
                if(data_array_x.size - index < 10):
                    size = data_array_x.size - index

                points_x = data_array_y[index: index+size]
                points_y = data_array_y[index: index + size]
                A = np.vstack([points_x, np.ones(size)]).T
                m, c = np.linalg.lstsq(A, points_y, rcond=None)[0]
                function = lambda x: m * x + c
                index = index + 10
                area = area + direction_list[i] * -1 * intergral.integrate(function, points_x[0], points_x[-1], 10)"""

        print(area)
        return np.float32(1.0)
    
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


def squareContour(n):
    size = int(n/2)
    x1 = np.linspace(2, 4 , num=size)
    y1 = np.ones(size) * 4
    c1 = np.stack((x1, y1), axis=1)

    y2 = np.linspace(2, 4 , num=size)
    x2 = np.ones(size) * 4
    c2 = np.stack((x2, np.flip(y2)), axis=1)

    y3 = np.ones(size) * 2
    c3 = np.stack((np.flip(x1), y3), axis=1)

    x4 = np.ones(size) * 2
    c4 = np.stack((x4, y2), axis=1)

    return np.concatenate((c1, c2, c3, c4))


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


if __name__ == "__main__":
    unittest.main()
    #ass4 = Assignment4()
    #ass4.area(squareContour)

