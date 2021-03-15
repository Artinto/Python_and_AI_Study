import numpy as np

A=[]

n,m=map(int,input().split())

for i in range(0,n):
    A.append(list(map(int,input().split())))

sum = np.sum(A, axis = 0)

print(np.prod(sum))
