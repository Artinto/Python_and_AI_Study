import numpy as np


n,m = map(int,input().split())
a = []
for i in range(n):# 2x2 행렬 만들기
    arr= list(map(int,input().split()))
    a.append(arr)


a=np.sum(a,axis=0)
print(np.prod(a))
