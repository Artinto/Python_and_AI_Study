import numpy as np

n=input()
arr1=np.array([], dtype=np.int8)
arr2=np.array([], dtype=np.int8)
for _ in range(int(n)):
    a=np.array(list(map(int, input().split())))
    arr1=np.concatenate((arr1, a))
for _ in range(int(n)):
    a=np.array(list(map(int, input().split())))
    arr2=np.concatenate((arr2, a))
arr1=arr1.reshape(int(n), int(n))
arr2=arr2.reshape(int(n), int(n))   
print(np.dot(arr1, arr2))
