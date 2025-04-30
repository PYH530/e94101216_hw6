import numpy as np

# 定義矩陣與向量
A = np.array([
    [1.19, 2.11, -100, 1],
    [14.2, -0.112, 12.2, -1],
    [0, 100, -99.9, 1],
    [15.3, 0.110, -13.1, -1]
], dtype=float)

b = np.array([1.12, 3.44, 2.15, 4.16], dtype=float)

# 拷貝，避免原始資料被破壞
A = A.copy()
b = b.copy()

n = len(b)

# Gaussian elimination with partial pivoting
for k in range(n - 1):
    max_row = np.argmax(np.abs(A[k:n, k])) + k
    if k != max_row:
        A[[k, max_row]] = A[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]

    if np.abs(A[k][k]) < 1e-12:
        raise ValueError(f"Zero or near-zero pivot encountered at row {k}")

    for i in range(k + 1, n):
        factor = A[i][k] / A[k][k]
        A[i, k:] = A[i, k:] - factor * A[k, k:]
        b[i] = b[i] - factor * b[k]

# Back substitution
x = np.zeros(n)
for i in range(n - 1, -1, -1):
    if np.abs(A[i][i]) < 1e-12:
        raise ValueError(f"Zero or near-zero diagonal value at row {i}")
    x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i][i]

print("Solution x =", x)
