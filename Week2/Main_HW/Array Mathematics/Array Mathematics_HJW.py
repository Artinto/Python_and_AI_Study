 import numpy

N,M=map(int,input().split())

A=numpy.array([list(map(int,input().split())) for n in range(N)])
B=numpy.array([list(map(int,input().split())) for n in range(N)])

print(numpy.add(A,B))
print(numpy.subtract(A,B))
print(numpy.multiply(A,B))
m=A/B

print(numpy.int_(m))

print(numpy.mod(A,B))
print(numpy.power(A,B))
