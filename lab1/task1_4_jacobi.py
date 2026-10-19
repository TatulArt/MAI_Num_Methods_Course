"""Задача 1.4: метод вращений Якоби для симметрической матрицы.

На каждой итерации обнуляется наибольший по модулю внедиагональный элемент a[i][j]
преобразованием подобия A' = U^T · A · U, где U — матрица вращения на угол phi:
    U[i][i] = U[j][j] = cos(phi),  U[i][j] = -sin(phi),  U[j][i] = sin(phi),
    tan(2·phi) = 2·a[i][j] / (a[i][i] - a[j][j]).
Накапливаем V = U_1 · U_2 · ... — столбцы V являются собственными векторами.
Остановка: sqrt(сумма квадратов внедиагональных элементов выше диагонали) < epsilon.
Запуск: python task1_4_jacobi.py [путь_к_json]   (по умолчанию data/task1_4.json)
"""

import math
import os
import sys

from matrix_utils import (load_json, identity, mat_mul, transpose, max_abs_diff,
                          fmt_matrix, fmt_vector, zeros)

MAX_ITERATIONS = 10000
SYMMETRY_TOLERANCE = 1e-12


def check_symmetric(matrix):
    size = len(matrix)

    if any(len(row) != size for row in matrix):
        raise ValueError("Матрица должна быть квадратной")

    for i in range(size):
        for j in range(i + 1, size):
            if abs(matrix[i][j] - matrix[j][i]) > SYMMETRY_TOLERANCE:
                raise ValueError("Матрица должна быть симметрической")


def off_diagonal_norm(matrix):
    size = len(matrix)
    total = 0.0

    for i in range(size):
        for j in range(i + 1, size):
            total += matrix[i][j] ** 2

    return math.sqrt(total)


def find_pivot(matrix):
    """Индексы (i, j), i < j, наибольшего по модулю внедиагонального элемента."""

    size = len(matrix)
    best_row, best_col = 0, 1

    for i in range(size):
        for j in range(i + 1, size):
            if abs(matrix[i][j]) > abs(matrix[best_row][best_col]):
                best_row, best_col = i, j

    return best_row, best_col


def rotation_matrix(matrix, row, col):
    size = len(matrix)
    difference = matrix[row][row] - matrix[col][col]

    if abs(difference) < 1e-15:
        angle = math.pi / 4
    else:
        angle = 0.5 * math.atan2(2 * matrix[row][col], difference)

    rotation = identity(size)
    rotation[row][row] = math.cos(angle)
    rotation[col][col] = math.cos(angle)
    rotation[row][col] = -math.sin(angle)
    rotation[col][row] = math.sin(angle)

    return rotation, angle


def jacobi(matrix, epsilon):
    """Возвращает (диагональная матрица, матрица собственных векторов V, число итераций, история)."""

    current = [[float(value) for value in row] for row in matrix]
    vectors = identity(len(matrix))
    history = []

    for iteration in range(MAX_ITERATIONS + 1):
        norm = off_diagonal_norm(current)

        if norm < epsilon:
            return current, vectors, iteration, history

        row, col = find_pivot(current)
        rotation, angle = rotation_matrix(current, row, col)
        history.append((iteration + 1, row, col, angle, norm))
        current = mat_mul(transpose(rotation), mat_mul(current, rotation))
        vectors = mat_mul(vectors, rotation)

    raise RuntimeError("Метод не сошёлся за допустимое число итераций")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "data", "task1_4.json")
    data = load_json(path)
    matrix, epsilon = data["matrix"], data["epsilon"]
    check_symmetric(matrix)
    size = len(matrix)
    print(f"Заданная точность ε = {epsilon}")
    print(fmt_matrix(matrix, "\nИсходная матрица A"))

    diagonalized, vectors, iterations, history = jacobi(matrix, epsilon)
    print("\nХод итераций (t = sqrt(сумма квадратов внедиагональных элементов) до вращения):")

    for iteration, row, col, angle, norm in history:
        print(f"  k={iteration:<3} t = {norm:.6f}  обнуляем a[{row + 1}][{col + 1}], угол phi = {angle:.6f} рад")

    print(f"\nКоличество итераций: {iterations}")
    print(f"Итоговое t = {off_diagonal_norm(diagonalized):.6f} < ε")
    print(fmt_matrix(diagonalized, "\nМатрица после вращений (почти диагональная)"))

    eigenvalues = [diagonalized[i][i] for i in range(size)]
    print("\nСобственные значения:\n" + fmt_vector(eigenvalues, "λ"))
    print(fmt_matrix(vectors, "\nМатрица собственных векторов V (столбцы — векторы)"))
    print("\nСобственные векторы:")

    for i in range(size):
        column = [vectors[row][i] for row in range(size)]
        print(f"  h{i + 1} = [" + "  ".join(f"{value:>10.6f}" for value in column) + " ]")

    lambda_matrix = zeros(size, size)

    for i in range(size):
        lambda_matrix[i][i] = eigenvalues[i]

    left_side = mat_mul(matrix, vectors)
    right_side = mat_mul(vectors, lambda_matrix)
    print(fmt_matrix(left_side, "\nA·V"))
    print(fmt_matrix(right_side, "\nV·Λ"))
    print(f"Проверка A·V = V·Λ: max|A·V - V·Λ| = {max_abs_diff(left_side, right_side):.3e}")
    print(f"Ортогональность: max|V^T·V - E| = {max_abs_diff(mat_mul(transpose(vectors), vectors), identity(size)):.3e}")


if __name__ == "__main__":
    main()
