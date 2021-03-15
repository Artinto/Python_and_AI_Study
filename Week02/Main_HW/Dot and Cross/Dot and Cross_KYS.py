import numpy

n = int(input())

A = [list(map(int, input().split())) for _ in range(n)]
B = [list(map(int, input().split())) for _ in range(n)]

A = numpy.array(A)
B = numpy.array(B)

print(numpy.dot(A, B))