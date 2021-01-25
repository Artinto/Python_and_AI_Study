import numpy

n, m = [int(i) for i in input().split()]
A = []

for i in range(n):
    A.append(input().split())

A = numpy.array(A, int)

A = numpy.sum(A, axis=0)
A = numpy.prod(A)
print(A)
