def integrate_function(func, a, b, n):
    delta_x = (b - a) / n
    result = 0
    x = a
    for i in range(n):
        result += func(x) * delta_x
        x += delta_x
    return result
