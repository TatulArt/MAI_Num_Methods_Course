"""Общие вспомогательные функции для лабораторной №1 (чистый Python, без NumPy)."""

import json


def load_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def identity(size):
    return [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]


def zeros(rows, cols):
    return [[0.0] * cols for _ in range(rows)]


def copy_matrix(matrix):
    return [row[:] for row in matrix]


def mat_mul(left, right):
    rows, inner, cols = len(left), len(right), len(right[0])

    return [[sum(left[i][k] * right[k][j] for k in range(inner)) for j in range(cols)] for i in range(rows)]


def mat_vec(matrix, vector):
    return [sum(element * coord for element, coord in zip(row, vector)) for row in matrix]


def permute_rows(matrix, permutation):
    """Строка i результата = строка permutation[i] исходной матрицы (это P·A)."""

    return [matrix[source][:] for source in permutation]


def permutation_matrix(permutation):
    size = len(permutation)
    result = zeros(size, size)

    for i, source in enumerate(permutation):
        result[i][source] = 1.0

    return result


def max_abs_diff(first, second):
    return max(abs(x - y) for row_first, row_second in zip(first, second) for x, y in zip(row_first, row_second))


def fmt_matrix(matrix, title=None, precision=6):
    lines = [title + ":"] if title else []

    for row in matrix:
        cells = "  ".join(f"{(0.0 if abs(value) < 5e-13 else value):>{precision + 6}.{precision}f}" for value in row)
        lines.append(f"  [{cells} ]")

    return "\n".join(lines)


def fmt_vector(vector, name="x", precision=6):
    return "\n".join(f"  {name}[{i + 1}] = {(0.0 if abs(value) < 5e-13 else value):.{precision}f}"
                     for i, value in enumerate(vector))


def vector_inf_norm(vector):
    return max(abs(value) for value in vector)


def matrix_inf_norm(matrix):
    return max(sum(abs(value) for value in row) for row in matrix)


def transpose(matrix):
    return [list(column) for column in zip(*matrix)]
