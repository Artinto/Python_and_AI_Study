import numpy as np

n,m=map(int,input().split())
list1=[]
for _ in range(n):
    l=list(map(int,input().split()))
    list1.append(l)
arr=np.array(list1)
arr=np.min(arr,axis=1)
print(np.max(arr))
