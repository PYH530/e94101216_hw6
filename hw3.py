import numpy as np

# 定義三對角矩陣 A 和向量 b
A = np.array([
    [3, -1, 0, 0],
    [-1, 3, -1, 0],
    [0, -1, 3, -1],
    [0, 0, -1, 3]
], dtype=float)

b = np.array([2, 3, 4, 1], dtype=float)

n = len(b)
L = np.zeros((n, n))
U = np.identity(n)  # 主對角線為 1

# Crout 分解適用 tridiagonal matrix
L[0, 0] = A[0, 0]
U[0, 1] = A[0, 1] / L[0, 0]

for i in range(1, n):
    L[i, i - 1] = A[i, i - 1]
    L[i, i] = A[i, i] - L[i, i - 1] * U[i - 1, i]
    if i < n - 1:
        U[i, i + 1] = A[i, i + 1] / L[i, i]

# 前代 Ly = b
y = np.zeros(n)
y[0] = b[0] / L[0, 0]
for i in range(1, n):
    y[i] = (b[i] - L[i, i - 1] * y[i - 1]) / L[i, i]

# 反代 Ux = y
x = np.zeros(n)
x[-1] = y[-1]
for i in range(n - 2, -1, -1):
    x[i] = y[i] - U[i, i + 1] * x[i + 1]

# 輸出解
np.set_printoptions(precision=4, suppress=True)
print("Crout 分解得到的解 x =\n", x)
