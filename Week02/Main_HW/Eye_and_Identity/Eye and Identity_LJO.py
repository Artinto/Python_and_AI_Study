import numpy as np

np.set_printoptions(legacy='1.13')
l=list(map(int,input().split()))
print(np.eye(l[0],l[1]))
