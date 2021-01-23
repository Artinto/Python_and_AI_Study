import numpy

n,m,p = map(int,input().split())
N=[]
M=[]

for i in range(0,n):
    N.append(list(map(int,input().split())))

for i in range(0,m):
    M.append(list(map(int,input().split())))

print(numpy.concatenate((N,M),axis = 0))
