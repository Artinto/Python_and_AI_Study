import numpy

arr=str(input())
arr=list(map(int,arr.split()))
print(numpy.reshape(arr,(3,3)))
