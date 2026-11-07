import sympy as sp


def real_solutions(equations, unknowns):
    """Решить систему и оставить изолированные действительные решения."""
    solutions = sp.solve(equations, unknowns, dict=True)
    result = []
    for solution in solutions:
        # Семейства решений нельзя незаметно заменить конечным списком точек.
        if any(v not in solution for v in unknowns):
            raise ValueError('Получено семейство решений; требуется отдельное исследование')
        point = {v: sp.simplify(solution[v]) for v in unknowns}
        if any(value.free_symbols for value in point.values()):
            raise ValueError('Получены параметрические решения')
        if any(value.is_real is False for value in point.values()):
            continue
        if any(value.is_real is not True for value in point.values()):
            raise ValueError('Не удалось установить действительность решения')
        if point not in result:
            result.append(point)
    return result


def solve_lagrange(f, g, variables):
    """Метод Лагранжа: две переменные, одно равенство g=0."""
    x, y = variables
    lam = sp.Symbol('lambda', real=True)
    f, g = sp.sympify(f), sp.sympify(g)
    print('1. Исходная задача')
    print('f =', f)
    print('g =', g, '= 0')

    # Нерегулярная точка должна одновременно лежать на ограничении.
    gradient_g = [sp.diff(g, v) for v in variables]
    irregular = real_solutions([g] + gradient_g, variables)
    print('\n2. Проверка регулярности')
    print('grad g =', gradient_g)
    print('Нерегулярные допустимые точки:', irregular or 'нет')
    if irregular:
        raise ValueError('Нерегулярные точки найдены; их нужно исследовать отдельно')

    # Самостоятельно составляем L и все необходимые условия метода.
    lagrangian = sp.expand(f + lam * g)
    unknowns = [x, y, lam]
    system = [sp.diff(lagrangian, v) for v in unknowns]
    print('\n3. Функция Лагранжа')
    print('L =', lagrangian)
    print('\n4. Частные производные и система')
    for variable, expression in zip(unknowns, system):
        print(f'dL/d{variable} = {expression} = 0')

    solutions = real_solutions(system, unknowns)
    if not solutions:
        raise ValueError('Изолированные действительные решения системы не найдены')

    # Вторые производные по x,y: исследуем L вдоль касательной к g=0.
    hessian = sp.Matrix([[sp.diff(lagrangian, u, v)
                          for v in variables] for u in variables])
    tangent = sp.Matrix([gradient_g[1], -gradient_g[0]])
    candidates = []
    print('\n5. Найденные точки, проверка и характер')
    for point in solutions:
        residuals = [sp.simplify(expr.subs(point)) for expr in system]
        if any(value != 0 for value in residuals):
            raise ArithmeticError('Решение не прошло точную проверку системы')
        if sp.simplify(g.subs(point)) != 0:
            raise ArithmeticError('Точка не удовлетворяет исходному ограничению')
        value = sp.simplify(f.subs(point))
        q = sp.simplify((tangent.T * hessian * tangent)[0].subs(point))
        if q.is_positive:
            kind = 'строгий локальный минимум'
        elif q.is_negative:
            kind = 'строгий локальный максимум'
        else:
            kind = 'второй порядок не даёт ответа'
        candidates.append({'point': point, 'value': value, 'q': q, 'kind': kind})
        print(f'x={point[x]}, y={point[y]}, lambda={point[lam]}; '
              f'f={value}; невязки={residuals}; q={q}; {kind}')

    # Это сравнение кандидатов; глобальность отдельно обосновывается в отчёте.
    values = [item['value'] for item in candidates]
    minimum = min(values, key=lambda v: float(v))
    maximum = max(values, key=lambda v: float(v))
    print('\n6. Сравнение значений в найденных точках')
    print('Наименьшее значение:', minimum)
    print('Наибольшее значение:', maximum)
    return {'lagrangian': lagrangian, 'system': system,
            'irregular': irregular, 'hessian': hessian,
            'candidates': candidates, 'minimum': minimum, 'maximum': maximum}


def main():
    print('Лабораторная работа 3. Борисюк Е. Г. Вариант 4\n')
    x, y = sp.symbols('x y', real=True)
    f = x * y
    g = x**2 + 4*y**2 - 8
    result = solve_lagrange(f, g, (x, y))
    print('\n7. Ответ для варианта 4')
    print('Глобальный минимум:', result['minimum'])
    print('Глобальный максимум:', result['maximum'])
    print('Глобальность следует из компактности эллипса и полноты списка точек;')
    print('подробное обоснование приведено в отчёте.')


if __name__ == '__main__':
    main()
