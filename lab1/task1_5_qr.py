"""Задача 1.5: QR-алгоритм нахождения собственных значений.

Разложение A = Q·R строится отражениями Хаусхолдера (Q ортогональна, R верхнетреугольная).
Итерации: A_0 = A, A_k = Q_k·R_k, A_{k+1} = R_k·Q_k (подобная матрица, спектр сохраняется).
Матрица A_k стремится к форме, где вещественным собственным значениям соответствуют
диагональные элементы, а комплексно-сопряжённой паре — блок 2x2 (его корни и есть пара).
NumPy используется только для проверки собственных значений (разрешено условием).
Запуск: python task1_5_qr.py [путь_к_json]   (по умолчанию data/task1_5.json)
"""

import cmath
import math
import os
import sys

from matrix_utils import load_json, identity, mat_mul, max_abs_diff, fmt_matrix

MAX_ITERATIONS = 10000


def householder_matrix(matrix, step):
    """Матрица отражения H = E - 2·v·v^T/(v^T·v), обнуляющая столбец step под диагональю."""

    size = len(matrix)
    column = [matrix[i][step] for i in range(step, size)]
    column_norm = math.sqrt(sum(value ** 2 for value in column))
    reflection = identity(size)

    if column_norm < 1e-15:
        return reflection

    direction = [value for value in column]
    direction[0] += (1.0 if column[0] >= 0 else -1.0) * column_norm
    direction_norm_squared = sum(value ** 2 for value in direction)

    for i in range(len(direction)):
        for j in range(len(direction)):
            reflection[step + i][step + j] -= 2 * direction[i] * direction[j] / direction_norm_squared

    return reflection


def qr_decompose(matrix):
    size = len(matrix)
    upper = [[float(value) for value in row] for row in matrix]
    orthogonal = identity(size)

    for step in range(size - 1):
        reflection = householder_matrix(upper, step)
        upper = mat_mul(reflection, upper)
        orthogonal = mat_mul(orthogonal, reflection)

    return orthogonal, upper


def extract_eigenvalues(matrix, epsilon):
    """Собственные значения по блочной структуре матрицы или None, если структура ещё не установилась.

    Столбец под диагональю мал (< epsilon) -> блок 1x1, значение = диагональный элемент.
    Иначе -> блок 2x2, его собственные значения — корни λ² - tr·λ + det = 0 (возможно, комплексные).
    """

    size = len(matrix)
    eigenvalues = []
    i = 0

    while i < size:
        below = max((abs(matrix[row][i]) for row in range(i + 1, size)), default=0.0)

        if below < epsilon:
            eigenvalues.append(complex(matrix[i][i]))
            i += 1
            continue

        if i == size - 1:
            return None

        below_block = max((abs(matrix[row][col]) for col in (i, i + 1) for row in range(i + 2, size)), default=0.0)

        if below_block >= epsilon:
            return None

        trace = matrix[i][i] + matrix[i + 1][i + 1]
        determinant = matrix[i][i] * matrix[i + 1][i + 1] - matrix[i][i + 1] * matrix[i + 1][i]
        root = cmath.sqrt(trace ** 2 - 4 * determinant)
        eigenvalues.append((trace + root) / 2)
        eigenvalues.append((trace - root) / 2)
        i += 2

    return eigenvalues


def qr_algorithm(matrix, epsilon):
    """Возвращает (итоговая матрица A_k, собственные значения, число итераций, история).
    Остановка: структура установилась и собственные значения между соседними итерациями меняются меньше epsilon.
    """

    current = [[float(value) for value in row] for row in matrix]
    previous_eigenvalues = None
    history = []

    for iteration in range(1, MAX_ITERATIONS + 1):
        orthogonal, upper = qr_decompose(current)
        current = mat_mul(upper, orthogonal)
        eigenvalues = extract_eigenvalues(current, epsilon)
        history.append((iteration, eigenvalues))

        if eigenvalues is not None and previous_eigenvalues is not None and len(eigenvalues) == len(previous_eigenvalues):
            change = max(abs(new - old) for new, old in zip(eigenvalues, previous_eigenvalues))

            if change < epsilon:
                return current, eigenvalues, iteration, history

        previous_eigenvalues = eigenvalues

    raise RuntimeError("QR-алгоритм не сошёлся за допустимое число итераций")


def format_complex(value, precision=6):
    if abs(value.imag) < 1e-12:
        return f"{value.real:.{precision}f}"

    sign = "+" if value.imag >= 0 else "-"

    return f"{value.real:.{precision}f} {sign} {abs(value.imag):.{precision}f}i"


def numpy_eigenvalues(matrix):
    """Эталон для проверки (в этой задаче NumPy разрешён)."""

    try:
        import numpy
    except ImportError:
        return None

    return list(numpy.linalg.eigvals(numpy.array(matrix, dtype=float)))


def solve_and_print(title, matrix, epsilon):
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(f"Заданная точность ε = {epsilon}")
    print(fmt_matrix(matrix, "\nИсходная матрица A"))

    orthogonal, upper = qr_decompose(matrix)
    print(fmt_matrix(orthogonal, "\nМатрица Q (первое разложение A = Q·R)"))
    print(fmt_matrix(upper, "\nМатрица R"))
    product = mat_mul(orthogonal, upper)
    print(fmt_matrix(product, "\nПроизведение Q·R"))
    print(f"max|Q·R - A| = {max_abs_diff(product, matrix):.3e}")

    final_matrix, eigenvalues, iterations, history = qr_algorithm(matrix, epsilon)
    print("\nХод итераций (оценки собственных значений):")

    for iteration, estimate in history:
        text = "структура ещё не установилась" if estimate is None else "  ".join(format_complex(value) for value in estimate)
        print(f"  k={iteration:<3} {text}")

    print(f"\nКоличество итераций: {iterations}")
    print(fmt_matrix(final_matrix, "Матрица A_k после итераций"))
    print("\nСобственные значения (QR-алгоритм):")

    for index, value in enumerate(eigenvalues):
        print(f"  λ{index + 1} = {format_complex(value)}")

    reference = numpy_eigenvalues(matrix)

    if reference is None:
        print("\nNumPy не установлен — сверка пропущена.")
        return

    print("\nПроверка NumPy (numpy.linalg.eigvals):")

    for index, value in enumerate(sorted(reference, key=lambda item: (-abs(item), -item.imag))):
        print(f"  λ{index + 1} = {format_complex(complex(value))}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "data", "task1_5.json")
    data = load_json(path)
    epsilon = data["epsilon"]
    solve_and_print("Матрица из условия задачи", data["matrix"], epsilon)
    print()
    solve_and_print("Собственный пример: матрица с комплексно-сопряжёнными собственными значениями",
                    data["custom_complex_matrix"], epsilon)


if __name__ == "__main__":
    main()
