import numpy as np

N = list(map(int, input().split()))
np.set_printoptions(legacy='1.13')
print(np.eye(*N))