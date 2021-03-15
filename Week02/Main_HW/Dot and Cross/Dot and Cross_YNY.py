import numpy
n = int(input())

A=[]
B=[]
for i in range (n):
    A.append(list(map(int, input().strip().split())))
    
for i in range (n):
    B.append(list(map(int, input().strip().split())))
    
A = numpy.array(A)
B = numpy.array(B)

print(numpy.dot(A, B))
