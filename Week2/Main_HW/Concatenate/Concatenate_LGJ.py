import numpy

n,m,p = map(int,input().split())
A=[]
B=[]

for i in range(0,n):
    A.append(list(map(int,input().split())))

for i in range(0,m):
    B.append(list(map(int,input().split())))

print(numpy.concatenate((A,B),axis = 0))
