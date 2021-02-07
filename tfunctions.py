import numpy as np

def f1(x):
    return 5

def f2(x):
    function = np.poly1d([1, -3, 5])
    return function(x)

def f3(x):
    function = np.poly1d([1, 0, 0])
    return np.sin(function(x))

def f4(x):
    function = np.poly1d([-2, 0, 0])
    return np.exp(function(x))

def f5(x):
    return np.arctan(x)

def f6(x):
    return np.sin(x) / x


def f7(x):
    return 1 / np.log(x)

def f8(x):
    return np.exp(np.exp(x))

def f9(x):
    return np.log(np.log(x))

def f10(x):
    return np.sin(np.log(x))

def f11(x):
    function = np.poly1d([2, 0, 0])
    function1 = lambda x: 1 / function(x)
    return (np.float32(2) ** function1(x)) * np.sin(1 / x)
