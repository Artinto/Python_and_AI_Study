import numpy as np

n, m = list(map(int, input().strip().split()))

arr_a, arr_b = [], []
for _ in range(n): arr_a.append(list(map(int, input().strip().split())))
for _ in range(n): arr_b.append(list(map(int, input().strip().split())))
arr_a, arr_b = np.array(arr_a), np.array(arr_b)

print(np.add(arr_a, arr_b))
print(np.subtract(arr_a, arr_b))
print(np.multiply(arr_a, arr_b))
print(arr_a//arr_b)
print(np.mod(arr_a, arr_b))
print(np.power(arr_a, arr_b))
