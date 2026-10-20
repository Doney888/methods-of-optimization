"""Метод ломаных: строка функции, отрезок, точность по значению функции."""
import ast
import heapq
import math
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt


def make_function(expression):
    # Разрешаем только арифметику, x и перечисленные математические функции.
    functions = {name: getattr(np, name) for name in
                 ('sin', 'cos', 'exp', 'log', 'sqrt', 'abs')}
    tree = ast.parse(expression, mode='eval')
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
               ast.Name, ast.Load, ast.Call, ast.Add, ast.Sub, ast.Mult,
               ast.Div, ast.Pow, ast.UAdd, ast.USub)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError('Недопустимая конструкция в строке функции')
        if isinstance(node, ast.Name) and node.id not in {'x', 'pi', 'e', *functions}:
            raise ValueError('Неизвестное имя: ' + node.id)
        if isinstance(node, ast.Constant) and (
                type(node.value) not in (int, float) or not math.isfinite(node.value)):
            raise ValueError('Нужны конечные вещественные константы')
        if isinstance(node, ast.Call) and (
                not isinstance(node.func, ast.Name) or node.func.id not in functions
                or len(node.args) != 1 or node.keywords):
            raise ValueError('У математической функции должен быть один аргумент')
    code = compile(tree, '<function>', 'eval')

    def f(x):
        with np.errstate(all='raise'):
            return eval(code, {'__builtins__': {}},
                        dict(functions, x=x, pi=np.pi, e=np.e))

    return f, tree.body


def lipschitz_bound(node, a, b):
    # Для каждого подвыражения возвращаем диапазон [lo, hi] и верхнюю оценку L.
    def bound(t):
        if isinstance(t, ast.Constant):
            return float(t.value), float(t.value), 0.0
        if isinstance(t, ast.Name):
            if t.id == 'x':
                return a, b, 1.0
            value = {'pi': math.pi, 'e': math.e}[t.id]
            return value, value, 0.0
        if isinstance(t, ast.UnaryOp):
            lo, hi, L = bound(t.operand)
            return (-hi, -lo, L) if isinstance(t.op, ast.USub) else (lo, hi, L)
        if isinstance(t, ast.Call):
            lo, hi, L = bound(t.args[0])
            name = t.func.id
            if name in ('sin', 'cos'):
                return -1.0, 1.0, L
            if name == 'exp':
                return math.exp(lo), math.exp(hi), math.exp(hi) * L
            if name == 'log' and lo > 0:
                return math.log(lo), math.log(hi), L / lo
            if name == 'sqrt' and lo > 0:
                return math.sqrt(lo), math.sqrt(hi), L / (2 * math.sqrt(lo))
            if name == 'abs':
                return (0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi)),
                        max(abs(lo), abs(hi)), L)
            raise ValueError('Для этого выражения задайте корректное L вручную')
        lo, hi, Lu = bound(t.left)
        low, high, Lv = bound(t.right)
        M, N = max(abs(lo), abs(hi)), max(abs(low), abs(high))
        if isinstance(t.op, ast.Add):
            return lo + low, hi + high, Lu + Lv
        if isinstance(t.op, ast.Sub):
            return lo - high, hi - low, Lu + Lv
        if isinstance(t.op, ast.Mult):
            values = [lo * low, lo * high, hi * low, hi * high]
            return min(values), max(values), N * Lu + M * Lv
        if isinstance(t.op, ast.Div) and not low <= 0 <= high:
            values = [lo / low, lo / high, hi / low, hi / high]
            m = min(abs(low), abs(high))
            return min(values), max(values), Lu / m + M * Lv / m**2
        if isinstance(t.op, ast.Pow) and Lv == 0 and low == high and low.is_integer():
            n = int(low)
            if n == 0:
                return 1.0, 1.0, 0.0
            if n > 0:
                values = [lo**n, hi**n] + ([0.0] if lo <= 0 <= hi else [])
                return min(values), max(values), n * M**(n - 1) * Lu
            if not lo <= 0 <= hi:
                values = [lo**n, hi**n]
                return min(values), max(values), abs(n) * min(abs(lo), abs(hi))**(n-1) * Lu
        raise ValueError('Не удалось оценить L автоматически; задайте его вручную')

    L = bound(node)[2]
    if not math.isfinite(L):
        raise ValueError('Оценка L не является конечной')
    return max(L * (1 + 1e-12), 1e-12)


def intersection(xl, fl, xr, fr, L):
    # Пересечение двух опорных прямых внутри соседнего интервала.
    x = (xl + xr) / 2 - (fr - fl) / (2 * L)
    lower = (fl + fr - L * (xr - xl)) / 2
    return x, lower


def piyavskii(expression, a, b, eps=0.01, L=None, direction='min', max_iter=100000):
    if not (math.isfinite(a) and math.isfinite(b) and a < b):
        raise ValueError('Нужен конечный отрезок a < b')
    if not math.isfinite(eps) or eps <= 0 or max_iter <= 0:
        raise ValueError('eps и max_iter должны быть положительными')
    if direction not in ('min', 'max'):
        raise ValueError('direction должен быть min или max')
    f, tree = make_function(expression)
    L = lipschitz_bound(tree, a, b) if L is None else float(L)
    if not math.isfinite(L) or L <= 0:
        raise ValueError('L должно быть положительным и конечным')
    sign = 1 if direction == 'min' else -1

    def value(x):
        y = sign * float(f(x))
        if not math.isfinite(y):
            raise ValueError('Функция должна быть конечной на отрезке')
        return y

    started = perf_counter()
    fa, fb = value(a), value(b)
    samples = {a: fa, b: fb}
    x_best, upper = min(samples.items(), key=lambda p: p[1])
    heap, history = [], []

    def add_interval(xl, fl, xr, fr):
        if abs(fr - fl) > L * (xr - xl) + 1e-10 * max(1, abs(fl), abs(fr)):
            raise ValueError('Заданное L противоречит уже вычисленным значениям')
        x, lower = intersection(xl, fl, xr, fr, L)
        heapq.heappush(heap, (lower, xl, xr, fl, fr, x))

    add_interval(a, fa, b, fb)
    iteration = 0
    while True:
        lower = heap[0][0]
        history.append({'iteration': iteration, 'upper': upper,
                        'lower': lower, 'gap': upper - lower, 'x_best': x_best,
                        'candidate': heap[0][5], 'left': heap[0][1], 'right': heap[0][2]})
        # Остановка по доказанной оценке погрешности значения функции.
        if upper - lower <= eps:
            break
        if iteration >= max_iter:
            raise RuntimeError('Превышен max_iter, точность ещё не достигнута')
        _, xl, xr, fl, fr, x = heapq.heappop(heap)
        if not xl < x < xr or x in samples:
            raise RuntimeError('Достигнут предел машинной точности')
        fx = value(x)
        samples[x] = fx
        if fx < upper:
            x_best, upper = x, fx
        # Удалённый интервал заменяем двумя; остальные вершины сохраняются.
        add_interval(xl, fl, x, fx)
        add_interval(x, fx, xr, fr)
        iteration += 1

    elapsed = perf_counter() - started
    vertices = sorted(samples.items())
    return {'expression': expression, 'a': a, 'b': b, 'eps': eps, 'L': L,
            'direction': direction, 'x': x_best, 'fun': sign * upper,
            'lower': lower if sign == 1 else -upper,
            'upper': upper if sign == 1 else -lower,
            'gap': upper - lower, 'iterations': iteration,
            'evaluations': len(samples), 'seconds': elapsed,
            'samples': [(x, sign*y) for x, y in vertices], 'history': history}


def plot_result(result, path=None):
    f, _ = make_function(result['expression'])
    sign = 1 if result['direction'] == 'min' else -1
    points = [(x, sign*y) for x, y in result['samples']]
    # Точные вершины ломаной; сетка используется только для рисунка f(x).
    vx, vy = [points[0][0]], [points[0][1]]
    for (xl, fl), (xr, fr) in zip(points, points[1:]):
        x, y = intersection(xl, fl, xr, fr, result['L'])
        vx.extend([x, xr]); vy.extend([y, fr])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    grid = np.linspace(result['a'], result['b'], 2500)
    values = np.array([float(f(x)) for x in grid])
    for ax in axes:
        ax.plot(grid, values, color='#1d4f91', label='Исходная функция')
        ax.plot(vx, sign*np.array(vy), color='#d47a14', lw=0.9, label='Итоговая ломаная')
        ax.scatter(*zip(*result['samples']), s=7, color='#728195', alpha=0.35)
        ax.scatter([result['x']], [result['fun']], color='#bb2233', zorder=5, label='Найденный экстремум')
        ax.set_xlabel('x'); ax.set_ylabel('f(x)'); ax.grid(alpha=0.2)
    axes[0].set_title('Весь отрезок'); axes[0].legend(fontsize=8)
    width = min((result['b']-result['a'])/4, 0.15)
    axes[1].set_xlim(max(result['a'],result['x']-width), min(result['b'],result['x']+width))
    level = result['fun']
    axes[1].set_ylim(level-max(0.03,3*result['eps']), level+max(0.1,10*result['eps']))
    axes[1].set_title('Увеличение около найденной точки')
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches='tight')
    return fig


if __name__ == '__main__':
    result = piyavskii('x**2 + 10*(1-cos(2*pi*x))', -2.0, 2.0, 0.01)
    print({k: result[k] for k in ('x', 'fun', 'L', 'lower', 'upper', 'gap',
                                  'iterations', 'evaluations', 'seconds')})
    plot_result(result, 'visualization.png')
    plt.show()
