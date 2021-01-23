import numpy as np
n, m=input().split()
n=int(n)
m=int(m)
arr1=np.array([], dtype=np.int8)
arr2=np.array([], dtype=np.int8)

for _ in range(n):
    a=np.array(list(map(int, input().split())))
    arr1=np.concatenate((arr1, a))
for _ in range(n):
    a=np.array(list(map(int, input().split())))
    arr2=np.concatenate((arr2, a))

arr1=arr1.reshape((n, m))
arr2=arr2.reshape((n, m))
print(np.add(arr1, arr2))
print(np.subtract(arr1, arr2))
print(np.multiply(arr1, arr2))
print(arr1//arr2)
print(np.mod(arr1, arr2))
print(np.power(arr1, arr2))
