import numpy as np

l=list(map(int,input().split()))
arr=np.array(l)
print(np.reshape(arr,(3,3)))
