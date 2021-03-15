import numpy

N,M = input().split()

N = int(N)
M = int(M)

M_ = []
for _ in range(N):
    P = list(map(int, input().split()))
    M_.append(P)
     
M_ = numpy.min(M_, axis = 1)
M_ = numpy.max(M_)

print(M_)
