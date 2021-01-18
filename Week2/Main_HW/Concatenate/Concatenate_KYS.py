import numpy

n, m, p = map(int, input().split())

arr_a = [list(map(int, input().split())) for _ in range(n)]
arr_b = [list(map(int, input().split())) for _ in range(m)]

num_a = numpy.array(arr_a)
num_b = numpy.array(arr_b)

print(numpy.concatenate((num_a, num_b), axis=0))