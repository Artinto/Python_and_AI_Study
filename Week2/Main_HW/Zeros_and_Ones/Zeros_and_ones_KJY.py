import numpy

a=list(map(int,input().split(' ')))
n=a.count(a)
t=tuple(a)
print(numpy.zeros(t,dtype=numpy.int))
print(numpy.ones(t,dtype=numpy.int))
