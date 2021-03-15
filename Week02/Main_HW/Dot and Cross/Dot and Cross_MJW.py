import numpy as np

n = int(input().strip())

arr_a, arr_b = [], []
for _ in range(n): arr_a.append(list(map(int, input().strip().split())))
for _ in range(n): arr_b.append(list(map(int, input().strip().split())))
arr_a, arr_b = np.array(arr_a), np.array(arr_b)

print(np.dot(arr_a, arr_b))
