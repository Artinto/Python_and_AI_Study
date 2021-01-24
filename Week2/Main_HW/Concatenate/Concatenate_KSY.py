import numpy
a, b, c = map(int,input().split())
A=[]
B=[]
for i in range(a):
    n=list(map(int,input().split()))
    A.append(n)
for i in range(b):
    n=list(map(int,input().split()))
    B.append(n)
arr1=numpy.array(A)
arr2=numpy.array(B)
print(numpy.concatenate((arr1,arr2),axis=0))


