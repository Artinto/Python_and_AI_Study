import numpy

N,M = input().split()
N = int(N)

P = []
for _ in range(N):
    M_ = list(map(int,input().split()))
    P.append(M_)       

P = numpy.sum(P, axis = 0)
print(numpy.prod(P))
