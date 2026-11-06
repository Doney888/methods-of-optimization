"""Лабораторная №4. Динамическое программирование, вариант 4."""


class ResourceAllocation:
    """Распределение целого ресурса между независимыми сервисами."""

    def __init__(self, total, profits, lower, upper):
        self.total = total
        self.profits = profits
        self.lower = lower
        self.upper = upper
        self.n = len(profits)
        self.values = []
        self.choices = []

    def solve(self):
        """Обратный проход: values[k][s] для сервисов k, ..., n-1."""
        self.values = [[float('-inf')] * (self.total + 1)
                       for _ in range(self.n + 1)]
        self.choices = [[[] for s in range(self.total + 1)]
                        for k in range(self.n)]
        self.values[self.n][0] = 0

        # Следующие этапы считаются раньше текущего.
        for k in range(self.n - 1, -1, -1):
            for s in range(self.total + 1):
                for u in range(self.lower[k], min(self.upper[k], s) + 1):
                    remaining = self.values[k + 1][s - u]
                    # Исключаем остатки, которые следующие сервисы не могут распределить.
                    if remaining == float('-inf'):
                        continue
                    candidate = self.profits[k][u] + remaining
                    if candidate > self.values[k][s]:
                        self.values[k][s] = candidate
                        self.choices[k][s] = [u]
                    elif candidate == self.values[k][s]:
                        # При равенстве сохраняем все оптимальные управления.
                        self.choices[k][s].append(u)
        return self.values[0][self.total]

    def restore(self, k=0, s=None):
        """Прямой проход по всем сохранённым оптимальным управлениям."""
        if s is None:
            s = self.total
        if k == self.n:
            return [()] if s == 0 else []
        plans = []
        for u in self.choices[k][s]:
            # Строим все оптимальные продолжения после текущего решения.
            for tail in self.restore(k + 1, s - u):
                plans.append((u,) + tail)
        return plans


# Данные варианта отделены от общего алгоритма.
TOTAL = 7
NAMES = ['Обработка заказов', 'Аналитика', 'Поиск']
PROFITS = [
    [0, 6, 10, 13, 16, 18, 19, 20],
    [0, 2, 7, 12, 16, 19, 21, 23],
    [0, 4, 9, 13, 15, 17, 18, 19],
]
LOWER = [1, 0, 0]
UPPER = [3, 3, 3]


def show_result(problem):
    print('s:', *range(problem.total + 1))
    for k in range(problem.n - 1, -1, -1):
        row = ['-' if x == float('-inf') else str(x)
               for x in problem.values[k]]
        print(f'F{k + 1}:', ' '.join(row))
        print(f'U{k + 1}:', problem.choices[k])
    print('Максимальная производительность:', problem.values[0][problem.total])
    print('Оптимальные распределения (заказы, аналитика, поиск):')
    for plan in problem.restore():
        parts = [problem.profits[k][u] for k, u in enumerate(plan)]
        print(plan, '->', ' + '.join(map(str, parts)), '=', sum(parts))


def main():
    problem = ResourceAllocation(TOTAL, PROFITS, LOWER, UPPER)
    problem.solve()
    show_result(problem)


if __name__ == '__main__':
    main()
