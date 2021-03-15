import numpy as np

n,m=map(int,input().split())
list1=[]
list2=[]
for _ in range(n):
    l1=list(map(int,input().split()))
    list1.append(l1)
for _ in range(n):
    l2=list(map(int,input().split()))
    list2.append(l2)
arr1=np.array(list1)
arr2=np.array(list2)
arr1=np.reshape(arr1,(n,m))
arr2=np.reshape(arr2,(n,m))

print(np.add(arr1,arr2))
print(np.subtract(arr1,arr2))
print(np.multiply(arr1,arr2))
print(arr1//arr2)
print(np.mod(arr1,arr2))
print(np.power(arr1,arr2))
