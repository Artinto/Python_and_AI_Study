import numpy as np

n, m = list(map(int, input().strip().split()))

arr = []
for _ in range(n):
    arr.append(list(map(int, input().strip().split())))
arr = np.array(arr)

# sum
x = np.sum(arr, axis=0)

# product
print(np.prod(x))
