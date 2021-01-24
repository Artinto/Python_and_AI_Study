import numpy as np


n=int(input())
a = []
b= []
for i in range(n):# 2x2 행렬 만들기
    arr1= list(map(int,input().split()))
    a.append(arr1)
for i in range(n):# 2x2 행렬 만들기
    arr2= list(map(int,input().split()))
    b.append(arr2)
print(np.dot(a,b))
