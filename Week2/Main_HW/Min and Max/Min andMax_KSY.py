import numpy as np


n,m = map(int,input().split())
a = []
for i in range(n):# 2x2 행렬 만들기
    arr= list(map(int,input().split()))
    a.append(arr)
b=np.min(a,axis=1)
print(np.max(b))
