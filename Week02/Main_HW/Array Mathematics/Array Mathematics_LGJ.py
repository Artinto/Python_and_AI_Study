import numpy as np

A=[]
B=[]

n,m=map(int,input().split())

for i in range(0,n):
    A.append(list(map(int,input().split())))

for i in range(0,n):
    B.append(list(map(int,input().split())))

Aarr = np.array(A, int)
Barr = np.array(B, int)

print(Aarr + Barr)
print(Aarr - Barr)
print(Aarr * Barr)
print(np.floor_divide(Aarr, Barr))
print(Aarr % Barr)
print(Aarr ** Barr)
