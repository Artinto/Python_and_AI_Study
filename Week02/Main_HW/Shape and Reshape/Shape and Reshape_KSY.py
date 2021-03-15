import numpy
A=[]
n=list(map(int,input().split()))
A.append(n)
A_array = numpy.array(A)
print(numpy.reshape(A_array,(3,3)))
