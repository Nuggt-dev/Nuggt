def integrate(func, a, b, n):
    h = (b - a) / n
    s = (func(a) + func(b)) / 2
    for i in range(1, n):
        s += func(a + i * h)
    return h * s
