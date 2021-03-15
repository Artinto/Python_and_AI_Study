import numpy



N,M = map(int,input().split())

arr = []
for _ in range(N):
    arr.append(list(map(int,input().split())))       

arr = numpy.sum(arr, axis = 0)
print(numpy.prod(arr))
