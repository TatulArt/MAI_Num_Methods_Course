"""Задача 1.2: метод прогонки (алгоритм Томаса) для трёхдиагональной СЛАУ.

Матрица не хранится целиком: используются только три диагонали.
Запуск: python task1_2_thomas.py [путь_к_json]   (по умолчанию data/task1_2.json)
"""

import os
import sys

from matrix_utils import load_json, fmt_vector


def check_input(lower_diagonal, main_diagonal, upper_diagonal, rhs):
    size = len(main_diagonal)

    if size < 2:
        raise ValueError("Размерность системы должна быть не меньше 2")

    if len(lower_diagonal) != size - 1 or len(upper_diagonal) != size - 1 or len(rhs) != size:
        raise ValueError("Длины диагоналей: main = n, lower = upper = n-1, rhs = n")


def is_diagonally_dominant(lower_diagonal, main_diagonal, upper_diagonal):
    """Достаточное условие устойчивости прогонки: |b_i| >= |a_i| + |c_i| (и строгое хотя бы в одной строке)."""

    size = len(main_diagonal)
    strict_found = False

    for i in range(size):
        off_diagonal_sum = 0.0

        if i > 0:
            off_diagonal_sum += abs(lower_diagonal[i - 1])

        if i < size - 1:
            off_diagonal_sum += abs(upper_diagonal[i])

        if abs(main_diagonal[i]) < off_diagonal_sum:
            return False

        if abs(main_diagonal[i]) > off_diagonal_sum:
            strict_found = True

    return strict_found


def thomas_coefficients(lower_diagonal, main_diagonal, upper_diagonal, rhs):
    """Прямой ход: возвращает прогоночные коэффициенты alpha, beta и знаменатели denominators.

    x_i = alpha_i · x_{i+1} + beta_i, где
        denominator_i = b_i + a_i · alpha_{i-1}
        alpha_i = -c_i / denominator_i
        beta_i = (d_i - a_i · beta_{i-1}) / denominator_i
    Для первой строки a_1 = 0, для последней c_n = 0 (alpha_n = 0).
    """

    size = len(main_diagonal)
    alpha = [0.0] * size
    beta = [0.0] * size
    denominators = [0.0] * size

    for i in range(size):
        lower_value = lower_diagonal[i - 1] if i > 0 else 0.0
        upper_value = upper_diagonal[i] if i < size - 1 else 0.0
        previous_alpha = alpha[i - 1] if i > 0 else 0.0
        previous_beta = beta[i - 1] if i > 0 else 0.0
        denominators[i] = main_diagonal[i] + lower_value * previous_alpha

        if abs(denominators[i]) < 1e-12:
            raise ValueError(f"Нулевой знаменатель в строке {i + 1}: метод прогонки неприменим")

        alpha[i] = -upper_value / denominators[i] + 0.0  # + 0.0 убирает «-0.0» в последней строке
        beta[i] = (rhs[i] - lower_value * previous_beta) / denominators[i]

    return alpha, beta, denominators


def thomas_back_substitution(alpha, beta):
    """Обратный ход: x_n = beta_n, затем x_i = alpha_i · x_{i+1} + beta_i."""

    size = len(alpha)
    solution = [0.0] * size
    solution[size - 1] = beta[size - 1]

    for i in range(size - 2, -1, -1):
        solution[i] = alpha[i] * solution[i + 1] + beta[i]

    return solution


def residual_norm(lower_diagonal, main_diagonal, upper_diagonal, rhs, solution):
    """max|Ax - d|, считается по трём диагоналям, без построения матрицы."""

    size = len(main_diagonal)
    worst = 0.0

    for i in range(size):
        value = main_diagonal[i] * solution[i]

        if i > 0:
            value += lower_diagonal[i - 1] * solution[i - 1]

        if i < size - 1:
            value += upper_diagonal[i] * solution[i + 1]

        worst = max(worst, abs(value - rhs[i]))

    return worst


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "data", "task1_2.json")
    data = load_json(path)
    lower_diagonal = data["lower_diagonal"]
    main_diagonal = data["main_diagonal"]
    upper_diagonal = data["upper_diagonal"]
    rhs = data["rhs"]
    check_input(lower_diagonal, main_diagonal, upper_diagonal, rhs)

    print("Нижняя диагональ a (n-1): ", lower_diagonal)
    print("Главная диагональ b (n):  ", main_diagonal)
    print("Верхняя диагональ c (n-1):", upper_diagonal)
    print("Правая часть d (n):       ", rhs)

    if not is_diagonally_dominant(lower_diagonal, main_diagonal, upper_diagonal):
        print("\nВНИМАНИЕ: диагонального преобладания нет, устойчивость прогонки не гарантирована.")

    alpha, beta, denominators = thomas_coefficients(lower_diagonal, main_diagonal, upper_diagonal, rhs)
    print("\nПрогоночные коэффициенты:")
    print(f"  {'i':>2} {'b_i + a_i*alpha_(i-1)':>24} {'alpha_i':>14} {'beta_i':>14}")

    for i in range(len(main_diagonal)):
        print(f"  {i + 1:>2} {denominators[i]:>24.6f} {alpha[i]:>14.6f} {beta[i]:>14.6f}")

    solution = thomas_back_substitution(alpha, beta)
    print("\nРешение системы:\n" + fmt_vector(solution))
    print(f"Невязка max|Ax - d| = {residual_norm(lower_diagonal, main_diagonal, upper_diagonal, rhs, solution):.3e}")


if __name__ == "__main__":
    main()
