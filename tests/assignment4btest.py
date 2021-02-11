import unittest
from sampleFunctions import *
from tqdm import tqdm
from assignment4B import Assignment4

def squareContour(n):
    size = int(n/4)
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

    def test_circle_area_from_contour(self):
        circ = Circle(cx=1, cy=1, radius=1, noise=0.0)
        ass4 = Assignment4()
        T = time.time()
        a_computed = ass4.area(contour=circ.contour, maxerr=0.1)
        T = time.time() - T
        a_true = circ.area()
        self.assertLess(abs((a_true - a_computed)/a_true), 0.1)

if __name__ == "__main__":
    unittest.main()

