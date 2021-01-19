import numpy

n, m = map(int, input().split())

arr = [list(map(int, input().split())) for _ in range(n)]

min_arr = numpy.min(arr, axis=1)
print(numpy.max(min_arr))


