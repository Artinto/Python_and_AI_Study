import numpy as np

n,m,p=map(int,input().split())
list1=[]
list2=[]
for _ in range(n):
    l1=list(map(int,input().split()))
    list1.append(l1)
for _ in range(m):
    l2=list(map(int,input().split()))
    list2.append(l2)
arr1=np.array(list1)
arr2=np.array(list2)
print(np.concatenate((arr1,arr2),axis=0))
