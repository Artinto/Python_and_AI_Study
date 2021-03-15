import numpy as np

n = int(input())

mat1 = list(map(int, input().split()))
mat1 = np.array([mat1])

for _ in range(n-1):
    line = list(map(int, input().split()))
    line = np.array([line])
    mat1 = np.concatenate((mat1, line))

mat2 = list(map(int, input().split()))
mat2 = np.array([mat2])

for _ in range(n-1):
    line = list(map(int, input().split()))
    line = np.array([line])
    mat2 = np.concatenate((mat2, line))


print(np.dot(mat1, mat2))