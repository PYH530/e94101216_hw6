import numpy as np

A = np.array([
    [4, 1, -1, 0],
    [1, 3, -1, 0],
    [-1, -1, 6, 2],
    [0, 0, 2, 5]
])

# 計算反矩陣
A_inv = np.linalg.inv(A)

# 印出反矩陣
print("反矩陣為：")
print(A_inv)
