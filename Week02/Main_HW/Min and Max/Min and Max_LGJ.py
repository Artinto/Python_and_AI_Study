import numpy as np

A=[]
n , m = map(int,input().split())

for i in range(0,n):
    A.append(list(map(int,input().split())))

min = np.min(A , axis=1)

print(np.max(min))
