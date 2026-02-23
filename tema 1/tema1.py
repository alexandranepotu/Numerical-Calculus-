import math
import time
import random

# ============================================================
# PROBLEMA 1 – Precizia masinii
# ============================================================

def find_machine_precision():
    print("Problema 1: Precizia masinii\n")

    m = 0
    while True:
        u = 10.0 ** (-m)
        if 1.0 + u == 1.0:
            break
        m += 1

    u = 10.0 ** (-(m - 1))

    print(f"m = {m-1}")
    print(f"u = 10^(-{m-1}) = {u:.16e}")
    print(f"1.0 + u = {1.0 + u:.16e}\n")

    return u


# ============================================================
# PROBLEMA 2 – Neasociativitate
# ============================================================

def test_addition_non_associativity(u):
    print("Problema 2a: Neasociativitatea adunarii\n")

    x = 1.0
    y = u / 10
    z = u / 10

    left = (x + y) + z
    right = x + (y + z)

    print(f"(x + y) + z = {left:.16e}")
    print(f"x + (y + z) = {right:.16e}")
    print(f"Egalitate? {left == right}\n")


def test_multiplication_non_associativity():
    print("Problema 2b: Neasociativitatea inmultirii\n")

    x = 1e308
    y = 1e308
    z = 1e-308

    left = (x * y) * z
    right = x * (y * z)

    print(f"(x * y) * z = {left}")
    print(f"x * (y * z) = {right}")
    print(f"Egalitate? {left == right}\n")


# ============================================================
# PROBLEMA 3 – Tangenta
# ============================================================

def tan_cf_lentz(x, epsilon):

    if abs(x) < 1e-12:
        return x

    tiny = 1e-12
    a = -x * x

    # b0 = 1
    f = 1.0
    C = f
    D = 0.0

    j = 1
    while True:
        b = 2 * j + 1

        D = b + a * D
        if D == 0.0:
            D = tiny
        D = 1.0 / D

        C = b + a / C
        if C == 0.0:
            C = tiny

        delta = C * D
        f *= delta

        if abs(delta - 1.0) < epsilon:
            break

        j += 1

    return x / f


def tan_polynomial(x):
    c1 = 1.0 / 3.0
    c2 = 2.0 / 15.0
    c3 = 17.0 / 315.0
    c4 = 62.0 / 2835.0

    x2 = x * x
    x3 = x2 * x

    return (x +
            c1 * x3 +
            c2 * x3 * x2 +
            c3 * x3 * x2 * x2 +
            c4 * x3 * x2 * x2 * x2)


def reduce_argument(x):
    pi = math.pi
    pi2 = pi / 2

    x = x % pi
    if x > pi2:
        x -= pi

    if abs(abs(x) - pi2) < 1e-10:
        return None

    return x


def my_tan_cf(x, epsilon):
    xr = reduce_argument(x)
    if xr is None:
        return float('inf')
    return tan_cf_lentz(xr, epsilon)


def my_tan_poly(x):
    xr = reduce_argument(x)
    if xr is None:
        return float('inf')

    pi4 = math.pi / 4
    if abs(xr) <= pi4:
        return tan_polynomial(xr)
    else:
        return math.copysign(1.0, xr) / tan_polynomial(math.pi / 2 - abs(xr))


# ============================================================
# COMPARATIE
# ============================================================

def compare_methods():
    print("Problema 3: Comparatie metode\n")

    values = [random.uniform(-1.5, 1.5) for _ in range(10000)]

    start = time.time()
    err_cf = []
    for x in values:
        err_cf.append(abs(math.tan(x) - my_tan_cf(x, 1e-10)))
    time_cf = time.time() - start

    start = time.time()
    err_poly = []
    for x in values:
        err_poly.append(abs(math.tan(x) - my_tan_poly(x)))
    time_poly = time.time() - start

    print("Fractii continue:")
    print(f"  timp = {time_cf:.6f}s")
    print(f"  eroare medie = {sum(err_cf)/len(err_cf):.2e}\n")

    print("Polinom:")
    print(f"  timp = {time_poly:.6f}s")
    print(f"  eroare medie = {sum(err_poly)/len(err_poly):.2e}\n")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    u = find_machine_precision()
    test_addition_non_associativity(u)
    test_multiplication_non_associativity()
    compare_methods()