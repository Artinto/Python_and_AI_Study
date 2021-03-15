import numpy

n,m=map(int,input().split(' '))
numpy.set_printoptions(legacy='1.13') #space 출력하도록 만듦
print(numpy.eye(n,m,k=0))
