import numpy as np

n, m = list(map(int, input().strip().split()))

arr = []
for _ in range(n):
    arr.append(list(map(int, input().strip().split())))
arr = np.array(arr)

# min
x = np.min(arr, axis=1)

# max
print(np.max(x))
