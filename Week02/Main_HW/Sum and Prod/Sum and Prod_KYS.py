import numpy

n, m = map(int, input().split())

arr = [list(map(int, input().split())) for _ in range(n)]
arr = numpy.array(arr)
print(numpy.prod(numpy.sum(arr, axis=0)))