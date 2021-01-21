import numpy

a=list(map(int,input().split(' ')))
t=tuple(a)
print(numpy.zeros(t,dtype=numpy.int))
print(numpy.ones(t,dtype=numpy.int))
