import numpy

a=input().split()
a=list(map(int, a))
Zeros=numpy.zeros((a), dtype=numpy.int)
Ones=numpy.ones((a), dtype=numpy.int)
print(Zeros)
print(Ones)
