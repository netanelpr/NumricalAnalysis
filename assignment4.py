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

        def arrage_to_parts(sample_data):
            poly_list = []
            current_list = []
            direction = 1
            iterator = ordered_sample_array
            prev_point = iterator()
            current_list.append(prev_point)
            current_list.append(iterator())
            if(prev_point[0] > current_list[1][0]):
                direction = -1

            for point in iterator:
                if(direction == 1):
                    if(point[0] < prev_point[0]):
                        poly_list.append(current_list)
                        current_list = [point]
                        direction = -1
                    else:
                        current_list.append(point)
                else:
                    if(point[0] > prev_point[0]):
                        poly_list.append(current_list)
                        current_list = [point]
                        direction = 1
                    else:
                        current_list.append(point)
                prev_point = point
            return poly_list               
        
        def sort_clockwise(sample_data):
            mean = np.mean(sample_data, axis=0)
            angles = np.arctan2((sample_data - mean)[:, 1], (sample_data - mean)[:, 0])
            angles[angles < 0] = angles[angles < 0] + 2 * np.pi
            sorting_indices = np.flip(np.argsort(angles))
            angles =angles[sorting_indices]
            sample_data = sample_data[sorting_indices]
            p_x = [point[0] for point in sample_data]
            p_y = [point[1] for point in sample_data]
            plt.scatter(p_x, p_y,c=[str(x) for x in np.arange(len(p_y)) / len(p_y)],cmap='gray')
            plt.show()

            return (sample_data, angles) 

        # replace these lines with your solution
        result = MyShape()
        sample_data = np.zeros((1000,2))
        for i in range(1000):
            sampl = sample()
            sample_data[i][0] = sampl[0]
            sample_data[i][1] = sampl[1]

        sorted_clockwise = sort_clockwise(sample_data)
        """print(sorted_clockwise)
        p_x = [point[0] for point in sorted_clockwise]
        p_y = [point[1] for point in sorted_clockwise]
        #plt.plot(p_x, p_y)
        #plt.show()
        plt.scatter(p_x, p_y,c=[str(x) for x in np.arange(len(p_y)) / len(p_y)],cmap='gray')
        plt.show()"""
        return result


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    #unittest.main()
    circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
    ass4 = Assignment4()
    shape = ass4.fit_shape(sample=circ, maxtime=30)
    shape.area()

