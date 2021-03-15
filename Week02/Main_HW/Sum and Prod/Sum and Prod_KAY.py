import numpy as np
n, m=input().split()
arr1=np.array([], dtype=np.int8)

for _ in range(int(n)):
    a=np.array(list(map(int, input().split())))
    arr1=np.concatenate((arr1, a))
arr1=arr1.reshape(int(n), int(m))
arr1=np.sum(arr1, axis=0)
print(np.prod(arr1))  
