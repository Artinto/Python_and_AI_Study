import numpy as np

n, m, p = map(int, input().split())

first_line = list(map(int, input().split()))
first_line = np.array([first_line])

for _ in range(n + m - 1):
    line = list(map(int, input().split()))
    line = np.array([line])
    first_line = np.concatenate((first_line, line), axis=0)

print(first_line)