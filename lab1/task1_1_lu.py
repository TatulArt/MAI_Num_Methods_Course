"""Задача 1.1: LU-разложение с выбором ведущего элемента (по столбцу).

Запуск: python task1_1_lu.py [путь_к_json]   (по умолчанию data/task1_1.json)
"""

import os
import sys

from matrix_utils import (load_json, identity, mat_mul, mat_vec, permute_rows,
                          permutation_matrix, max_abs_diff, fmt_matrix, fmt_vector)


def lu_decompose(matrix):
    """Возвращает (lower, upper, permutation, swaps_count): lower·upper = P·A,
    где строка i матрицы P·A = строка permutation[i] матрицы A.

    Алгоритм Гаусса с частичным выбором: на шаге k ищем в столбце k (строки k..size-1)
    элемент с наибольшим модулем, переставляем строки, затем зануляем элементы под диагональю;
    множители исключения сохраняем в lower.
    """

    size = len(matrix)
    upper = [[float(value) for value in row] for row in matrix]
    lower = [[0.0] * size for _ in range(size)]
    permutation = list(range(size))
    swaps_count = 0

    for k in range(size):
        pivot_row = max(range(k, size), key=lambda i: abs(upper[i][k]))

        if abs(upper[pivot_row][k]) < 1e-12:
            raise ValueError("Матрица вырождена: ведущий элемент равен нулю")

        if pivot_row != k:
            upper[k], upper[pivot_row] = upper[pivot_row], upper[k]
            lower[k], lower[pivot_row] = lower[pivot_row], lower[k]  # переставляем уже посчитанные множители
            permutation[k], permutation[pivot_row] = permutation[pivot_row], permutation[k]
            swaps_count += 1

        for i in range(k + 1, size):
            factor = upper[i][k] / upper[k][k]
            lower[i][k] = factor

            for j in range(k, size):
                upper[i][j] -= factor * upper[k][j]

    for i in range(size):
        lower[i][i] = 1.0

    return lower, upper, permutation, swaps_count


def solve_lu(lower, upper, permutation, rhs):
    """Решает Ax=b: L·y = P·b (прямой ход), U·x = y (обратный ход)."""

    size = len(lower)
    permuted_rhs = [rhs[permutation[i]] for i in range(size)]
    intermediate = [0.0] * size

    for i in range(size):
        intermediate[i] = permuted_rhs[i] - sum(lower[i][j] * intermediate[j] for j in range(i))

    solution = [0.0] * size

    for i in range(size - 1, -1, -1):
        solution[i] = (intermediate[i] - sum(upper[i][j] * solution[j] for j in range(i + 1, size))) / upper[i][i]

    return solution


def determinant(upper, swaps_count):
    """det A = (-1)^swaps_count · произведение диагонали upper (det L = 1)."""

    result = -1.0 if swaps_count % 2 else 1.0

    for i in range(len(upper)):
        result *= upper[i][i]

    return result


def inverse(lower, upper, permutation):
    """Решаем A·X = E по столбцам: каждый столбец обратной матрицы — решение Ax = e_i."""

    size = len(lower)
    eye = identity(size)
    columns = [solve_lu(lower, upper, permutation, [eye[row][col] for row in range(size)]) for col in range(size)]

    return [[columns[j][i] for j in range(size)] for i in range(size)]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "data", "task1_1.json")
    data = load_json(path)
    matrix, rhs = data["matrix"], data["rhs"]
    size = len(matrix)

    if any(len(row) != size for row in matrix) or len(rhs) != size:
        raise ValueError("Матрица должна быть квадратной, размер правой части должен совпадать с её размером")

    lower, upper, permutation, swaps_count = lu_decompose(matrix)
    print(fmt_matrix(matrix, "Исходная матрица A"))
    print("\nВектор b:\n" + fmt_vector(rhs, "b"))
    print("\nПерестановка строк (P·A: строка i = строка permutation[i] матрицы A):", [p + 1 for p in permutation])
    print(fmt_matrix(permutation_matrix(permutation), "Матрица перестановок P"))
    print("\n" + fmt_matrix(lower, "Матрица L"))
    print("\n" + fmt_matrix(upper, "Матрица U"))
    product = mat_mul(lower, upper)
    print("\n" + fmt_matrix(product, "Произведение L·U"))

    solution = solve_lu(lower, upper, permutation, rhs)
    print("\nРешение системы Ax = b:\n" + fmt_vector(solution))
    residual = max(abs(lhs - expected) for lhs, expected in zip(mat_vec(matrix, solution), rhs))
    print(f"Невязка max|Ax - b| = {residual:.3e}")

    inverse_matrix = inverse(lower, upper, permutation)
    print("\n" + fmt_matrix(inverse_matrix, "Обратная матрица A^-1"))
    print(f"\nОпределитель det A = {determinant(upper, swaps_count):.6f}  (число перестановок строк: {swaps_count})")

    check_identity = mat_mul(matrix, inverse_matrix)
    print("\n" + fmt_matrix(check_identity, "Проверка A·A^-1 (должна быть E)"))
    print(f"max|A·A^-1 - E| = {max_abs_diff(check_identity, identity(size)):.3e}")
    permuted_matrix = permute_rows(matrix, permutation)
    print("\n" + fmt_matrix(permuted_matrix, "P·A"))
    print(f"Проверка L·U = P·A: max|L·U - P·A| = {max_abs_diff(product, permuted_matrix):.3e}")


if __name__ == "__main__":
    main()
