import numpy

N, M = map(int, input().split())
S = []

for _ in range(N):
    s = list(map(int, input().split()))
    S.append(s)


arr = numpy.array(S)

min_ = numpy.min(arr, axis = 1)
max_ = numpy.max(min_, axis = 0)

print(max_)   
