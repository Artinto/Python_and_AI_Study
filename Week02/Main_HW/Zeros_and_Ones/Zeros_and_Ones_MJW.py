import numpy as np

x, y, z = list(map(int, input().strip().split()))

print(np.zeros((x, y, z), dtype = np.int))
print(np.ones((x, y, z), dtype = np.int))
