import numpy as np

n, _ = input().split()

n = int(n)

mat = list(map(int, input().split()))
mat = np.array([mat])

for _ in range(n-1):
    line = list(map(int, input().split()))
    line = np.array([line])
    mat = np.concatenate((mat, line))

print(np.prod(np.sum(mat, axis=0)))