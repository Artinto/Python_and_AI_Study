import numpy

n=int(input())

a1=numpy.array(list(map(int,input().split())))
for _ in range(n-1):
    new=numpy.array(list(map(int,input().split())))
    a1=numpy.concatenate((a1,new),axis=None)
a1=numpy.reshape(a1,(n,n))
a2=numpy.array(list(map(int,input().split())))
for _ in range(n-1):
    new=numpy.array(list(map(int,input().split())))
    a2=numpy.concatenate((a2,new),axis=None)
a2=numpy.reshape(a2,(n,n))

print(numpy.dot(a1,a2))
