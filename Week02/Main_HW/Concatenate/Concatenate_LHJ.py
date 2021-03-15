import numpy
n, m, p = input().split()

arr_a = []
arr_b = []

for i in range(int(n)):
    arr_a.append(input().split())

for i in range(int(m)):
    arr_b.append(input().split())

arr_a = numpy.array(arr_a, int)
arr_b = numpy.array(arr_b, int)

print(numpy.concatenate((arr_a, arr_b), axis=0))
