import numpy as np
np.set_printoptions(legacy='1.13')

n, m = list(map(int, input().strip().split()))

arr = np.eye(n, m)
print(arr)
