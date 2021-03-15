import numpy as np

N = int(input())
a = np.array(list(input().split() for _ in range(N)), dtype = int)
b = np.array(list(input().split() for _ in range(N)), dtype = int)
print(np.dot(a,b))
