import numpy as np

N=[]
M=[]

n,m=map(int,input().split())

for i in range(0,n):
    N.append(list(map(int,input().split())))

for i in range(0,n):
    M.append(list(map(int,input().split())))

narr = np.array(N, int)
marr = np.array(M, int)

print(narr + marr)
print(narr - marr)
print(narr * marr)
print(np.floor_divide(narr, marr))
print(narr % marr)
print(narr ** marr)
