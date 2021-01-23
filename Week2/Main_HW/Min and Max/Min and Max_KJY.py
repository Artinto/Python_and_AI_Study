import numpy

n,m=map(int,input().split())
first=list(map(int,input().split()))
for _ in range(n-1):
    add_arr=numpy.array(list(map(int,input().split())))
    first=numpy.concatenate((first,add_arr),axis=None)
arr=numpy.reshape(first,(n,m))

minis=numpy.min(arr,axis=1)
maxim=numpy.max(minis,axis=None)
print(maxim)
