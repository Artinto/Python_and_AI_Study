import numpy as np

N,M = map(int,input().split())
a = np.array(list(input().split() for _ in range(N)), dtype = int)
print(np.prod(np.sum(a, axis = 0)))
