import numpy as np

A=[]
B=[]

n = int(input())

for i in range(0,n):
    A.append(list(map(int,input().split())))

for i in range(0,n):
    B.append(list(map(int,input().split())))

print(np.dot(A,B))
