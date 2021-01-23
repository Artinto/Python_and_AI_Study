import numpy

n,m=map(int,input().split(' '))
list1=list(map(int,input().split(' ')))
for _ in range(n-1):
    tmp=list(map(int,input().split(' ')))
    list1=numpy.concatenate((list1,tmp),axis=0)
list1=numpy.reshape(list1,(n,m))
sum_array=numpy.sum(list1,axis=0)
print(numpy.product(sum_array))
