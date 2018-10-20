from sympy import Mul, Pow


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def div(a, b):
    return Mul(a, Pow(b, -1))


def inv(a):
    return Pow(a, -1)
