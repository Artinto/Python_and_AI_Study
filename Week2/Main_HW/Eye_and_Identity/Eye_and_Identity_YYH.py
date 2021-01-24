import numpy

numpy.set_printoptions(legacy='1.13')

a,b=map(int,input().split())

print (numpy.eye(a, b, k = 0))
