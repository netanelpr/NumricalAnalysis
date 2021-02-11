import unittest
from functionUtils import *
from tqdm import tqdm
import tfunctions
import matplotlib.pyplot as plt
from assignment1 import Assignment1

class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        index = 0
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200) * 20 - 10
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
            index = i
        mean_err = mean_err / 100

        print(index)
        T = time.time() - T
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)

    def test_1(self):
        self.tfunc("sqr", lambda x: x ** 2, -5, 5, 10)

    def test_2(self):
        self.tfunc("t2", lambda x: tfunctions.f2(x), -10, 10, 10)

    def test_3(self):
        self.tfunc("t3", lambda x: tfunctions.f3(x), -20, 20, 200)

    def test_7(self):
        self.tfunc("t7", lambda x: tfunctions.f7(x), 1.01, 5, 200)

    def test_12(self):
        self.tfunc("t_2_points", lambda x: x ** 2, -5, 5, 2)

    def test_13(self):
        self.tfunc("t_3_points", lambda x: x ** 2, -5, 5, 3)

    def test_14(self):
        self.tfunc("t_4_points", lambda x: x ** 2, -5, 5, 4)

    def tfunc(self, function_name, f, s, to, number_of_dots, draw=False):
        assignment1 = Assignment1()
        interpolated = assignment1.interpolate(f, s, to, number_of_dots)

        print(function_name)

        xs = (np.random.random(20)) * ((to - s)) - (to - s) / 2
        xs = np.concatenate((np.array([s, to]), xs))
        ys = []
        y2s = []
        r_err = 0
        for x in xs:
            if(x < s):
                x = s + 1

            y2 = interpolated(x)
            y2s.append(y2)
            y = f(x)
            ys.append(y)
            #print(x)
            #print(f"{x} {y} {y2} {abs((y - y2) / y)}")
            r_err = r_err + abs((y - y2) / y)

        if(draw):
            #x2s = [number for number in np.linspace(s * 1.0, to * 1.0)]
            #y2s = [f(number) for number in x2s]
            plt.plot(xs, y2s, "bo")
            plt.plot(xs, ys, "ro")
            plt.show()

        print(r_err / 20)

if __name__ == "__main__":
    unittest.main()