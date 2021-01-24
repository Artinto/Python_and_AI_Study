import numpy as np

arr = list(map(int, input().strip().split()))

np_arr = np.array(arr)
np_arr = np.reshape(np_arr, (3, 3))

print(np_arr)

