"""Задача 1.3: метод простых итераций и метод Зейделя.

Система Ax = b приводится к виду x = beta·x + c, где
    beta[i][j] = -a[i][j] / a[i][i]  (i != j),  beta[i][i] = 0,
    c[i] = b[i] / a[i][i].
Запуск: python task1_3_iterations.py [путь_к_json]   (по умолчанию data/task1_3.json)
"""

import os
import sys

from matrix_utils import load_json, matrix_inf_norm, vector_inf_norm, fmt_vector
from task1_1_lu import lu_decompose, solve_lu

MAX_ITERATIONS = 10000


def reduce_to_iterative_form(matrix, rhs):
    size = len(matrix)
    beta = [[0.0] * size for _ in range(size)]
    free_terms = [0.0] * size

    for i in range(size):
        if abs(matrix[i][i]) < 1e-12:
            raise ValueError(f"Нулевой элемент на диагонали в строке {i + 1}")

        free_terms[i] = rhs[i] / matrix[i][i]

        for j in range(size):
            if i != j:
                beta[i][j] = -matrix[i][j] / matrix[i][i]

    return beta, free_terms


def upper_part(beta):
    """Верхний треугольник матрицы beta (j > i) — нужен для оценки погрешности метода Зейделя."""

    size = len(beta)

    return [[beta[i][j] if j > i else 0.0 for j in range(size)] for i in range(size)]


def simple_iteration_step(beta, free_terms, previous):
    size = len(beta)

    return [free_terms[i] + sum(beta[i][j] * previous[j] for j in range(size)) for i in range(size)]


def seidel_step(beta, free_terms, previous):
    """Новые компоненты x_j (j < i) используются сразу, как только они вычислены."""

    size = len(beta)
    current = previous[:]

    for i in range(size):
        current[i] = free_terms[i] + sum(beta[i][j] * current[j] for j in range(size))

    return current


def iterate(step, beta, free_terms, error_factor, epsilon):
    """Общий цикл. Остановка по апостериорной оценке:
    ||x_k - x*|| <= error_factor · ||x_k - x_{k-1}|| < epsilon.
    Начальное приближение x_0 = c. Возвращает (решение, число итераций, история).
    """

    current = free_terms[:]
    history = [(0, current[:], None)]

    for iteration in range(1, MAX_ITERATIONS + 1):
        previous = current
        current = step(beta, free_terms, previous)
        difference = vector_inf_norm([new - old for new, old in zip(current, previous)])
        error_estimate = error_factor * difference
        history.append((iteration, current[:], error_estimate))

        if error_estimate < epsilon:
            return current, iteration, history

    raise RuntimeError("Метод не сошёлся за допустимое число итераций")


def print_history(title, history):
    print(f"\n{title}")

    for iteration, vector, error_estimate in history:
        values = "  ".join(f"{value:>11.6f}" for value in vector)
        error_text = f"{error_estimate:.6f}" if error_estimate is not None else "—"
        print(f"  k={iteration:<3} x = [{values} ]   оценка погрешности: {error_text}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "data", "task1_3.json")
    data = load_json(path)
    matrix, rhs, epsilon = data["matrix"], data["rhs"], data["epsilon"]
    size = len(matrix)

    if any(len(row) != size for row in matrix) or len(rhs) != size:
        raise ValueError("Матрица должна быть квадратной, размер правой части должен совпадать с её размером")

    beta, free_terms = reduce_to_iterative_form(matrix, rhs)
    beta_norm = matrix_inf_norm(beta)
    upper_norm = matrix_inf_norm(upper_part(beta))
    print(f"Заданная точность ε = {epsilon}")
    print(f"||beta||_inf = {beta_norm:.6f}  (условие сходимости: < 1)")

    if beta_norm >= 1:
        raise ValueError("||beta||_inf >= 1: достаточное условие сходимости не выполнено")

    print("Вектор c:\n" + fmt_vector(free_terms, "c"))
    simple_factor = beta_norm / (1 - beta_norm)
    seidel_factor = upper_norm / (1 - beta_norm)
    print(f"Множитель оценки (простые итерации) ||beta||/(1-||beta||)   = {simple_factor:.6f}")
    print(f"Множитель оценки (Зейдель)          ||C||/(1-||beta||)      = {seidel_factor:.6f}")

    simple_solution, simple_count, simple_history = iterate(simple_iteration_step, beta, free_terms, simple_factor, epsilon)
    seidel_solution, seidel_count, seidel_history = iterate(seidel_step, beta, free_terms, seidel_factor, epsilon)
    print_history("Метод простых итераций:", simple_history)
    print_history("Метод Зейделя:", seidel_history)

    lower, upper, permutation, _ = lu_decompose(matrix)
    reference = solve_lu(lower, upper, permutation, rhs)
    print("\nЭталонное решение (LU-разложение из задачи 1.1):\n" + fmt_vector(reference))
    print("\nМетод простых итераций:\n" + fmt_vector(simple_solution))
    print("\nМетод Зейделя:\n" + fmt_vector(seidel_solution))

    simple_error = vector_inf_norm([found - exact for found, exact in zip(simple_solution, reference)])
    seidel_error = vector_inf_norm([found - exact for found, exact in zip(seidel_solution, reference)])
    print("\nСравнение методов:")
    print(f"  Простые итерации: {simple_count} итераций, фактическая погрешность {simple_error:.3e}")
    print(f"  Зейдель:          {seidel_count} итераций, фактическая погрешность {seidel_error:.3e}")

    if seidel_count < simple_count:
        print(f"  Метод Зейделя сошёлся быстрее в {simple_count / seidel_count:.2f} раза.")
    elif seidel_count > simple_count:
        print(f"  Метод простых итераций сошёлся быстрее в {seidel_count / simple_count:.2f} раза.")
    else:
        print("  Количество итераций одинаковое.")


if __name__ == "__main__":
    main()
