from math import sqrt


def quadratic_roots(a, b, c):
    """Корни квадратного уравнения a*t^2 + b*t + c = 0."""
    d = b * b - 4 * a * c
    if d < 0:
        return []
    if d == 0:
        return [-b / (2 * a)]
    return [(-b - sqrt(d)) / (2 * a), (-b + sqrt(d)) / (2 * a)]


def lagrange():
    # Вариант 4: y+2*lambda*x=0; x+8*lambda*y=0; x^2+4*y^2=8.
    # Пара (0,0) недопустима, поэтому определитель 16*lambda^2-1=0.
    points = []
    for lam in quadratic_roots(16, 0, -1):
        ratio = -2 * lam  # y = ratio*x
        # Подставляем y=ratio*x в ограничение и вычисляем модуль x.
        x_abs = sqrt(8 / (1 + 4 * ratio * ratio))
        # Для каждого множителя восстанавливаем оба знака координаты x.
        for sign in (-1, 1):
            x = sign * x_abs
            y = ratio * x
            points.append((x, y, lam))
    return points


def main():
    points = lagrange()
    values = [x * y for x, y, lam in points]
    minimum, maximum = min(values), max(values)
    print('Лабораторная работа 3. Метод множителей Лагранжа')
    print('Борисюк Е. Г. Вариант 4')
    print('f(x,y) = x*y; x^2 + 4*y^2 = 8')
    print('L = x*y + lambda*(x^2 + 4*y^2 - 8)')
    print('Нерегулярных допустимых точек нет: (0,0) не лежит на эллипсе.')
    print('Определитель: 16*lambda^2 - 1 = 0')
    print('       x          y       lambda       f        невязка')
    for (x, y, lam), value in zip(points, values):
        # Невязка - наибольшая абсолютная ошибка трёх уравнений системы.
        residual = max(abs(y + 2*lam*x), abs(x + 8*lam*y),
                       abs(x*x + 4*y*y - 8))
        if residual >= 1e-10:
            raise ArithmeticError('Точка не удовлетворяет системе Лагранжа')
        kind = 'минимум' if abs(value-minimum) < 1e-10 else 'максимум'
        print(f'{x:10.6f} {y:10.6f} {lam:10.6f} {value:10.6f} {residual:.2e}  {kind}')
    print(f'Глобальный минимум: {minimum:.6f}')
    print(f'Глобальный максимум: {maximum:.6f}')


if __name__ == '__main__':
    main()

