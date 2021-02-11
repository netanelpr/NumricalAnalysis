import unittest
from sampleFunctions import *
from tqdm import tqdm
from assignment4A import Assignment4A

class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1,1,1))
        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    def test_delay(self):
        f = DELAYED(7)(NOISY(0.01)(poly(1,1,1)))

        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1,1,1)
        nf = NOISY(1)(f)
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse=0
        for x in np.linspace(0,1,1000):
            self.assertNotEqual(f(x), nf(x))
            mse+= (f(x)-ff(x))**2
        mse = mse/1000
        print(mse)

    def test_poly10(self):
        f = np.poly1d(np.random.randn(10))
        nf = NOISY(1)(f)
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse=0
        for x in np.linspace(0,1,1000):
            self.assertNotEqual(f(x), nf(x))
            mse+= (f(x)-ff(x))**2
        mse = mse/1000
        print(mse)

    def test_err_constant(self):
        f = poly(5)
        nf = NOISY(1)(f)
        self.mse_poly(f, nf, 0, 1)

    def test_err_2(self):
        f = poly(1, 1, 1)
        nf = NOISY(1)(f)
        self.mse_poly(f, nf, -6, 6)

    def test_err_linear_line(self):
        f = poly(5, 0.3)
        nf = NOISY(1)(f)
        self.mse_poly(f, nf, 0, 1)

    def test_err_cubic(self):
        f = poly(*np.random.random(4))
        nf = NOISY(1)(f)
        self.mse_poly(f, nf, 0, 1)

    def test_err_poly_4(self):
        f = poly(*np.random.random(5))
        nf = NOISY(1)(f)
        self.mse_poly(f, nf, 0, 1)

    def test_err_poly_15(self):
        f = poly(*np.random.random(16))
        nf = NOISY(1)(f)
        self.mse_poly(f, nf, 0, 1)

    def mse_poly(self, f, nf, a, b, maxtime=5):
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=a, b=b, d=max(10, f.order), maxtime=maxtime)
        T = time.time() - T
        mse=0
        for x in np.linspace(0,1,1000):
            self.assertNotEqual(f(x), nf(x))
            mse+= (f(x)-ff(x))**2
        mse = mse/1000
        print(f"time: {T}")
        print(f"mse: {mse}")
        print("")

if __name__ == "__main__":
    unittest.main()
