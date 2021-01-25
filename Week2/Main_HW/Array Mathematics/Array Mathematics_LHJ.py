import numpy

n, m = [int(i) for i in input().split()]

A = []
B = []

for i in range(n):
    A.append(input().split())

for i in range(n):
    B.append(input().split())

A = numpy.array(A, int)
B = numpy.array(B, int)

print(A+B, A-B, A*B, A//B, A%B, A**B, sep='\n')
