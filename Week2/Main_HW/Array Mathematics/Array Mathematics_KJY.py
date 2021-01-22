import numpy

a=[]
b=[]

n,m=map(int,input().split(' '))
for i in range(n):
    tmp=list(map(int,input().split(' ')))
    a.append(tmp)
for i in range(n):
    tmp=list(map(int,input().split(' ')))
    b.append(tmp)
a=numpy.array(a,float)
b=numpy.array(b,float)

my_array=numpy.add(a,b)
# 형 변환
my_array=numpy.array(my_array,dtype='int64')
# reshape 함수를 이용
print(numpy.reshape(my_array,(n,m)))

my_array=numpy.subtract(a,b)
# 형 변환
my_array=numpy.array(my_array,dtype='int64')
# reshape 함수를 이용
print(numpy.reshape(my_array,(n,m)))

my_array=numpy.multiply(a,b)
# 형 변환
my_array=numpy.array(my_array,dtype='int64')
# reshape 함수를 이용
print(numpy.reshape(my_array,(n,m)))

my_array=numpy.divide(a,b)
# 형 변환
my_array=numpy.array(my_array,dtype='int64')
# reshape 함수를 이용
print(numpy.reshape(my_array,(n,m)))

my_array=numpy.mod(a,b)
# 형 변환
my_array=numpy.array(my_array,dtype='int64')
# reshape 함수를 이용
print(numpy.reshape(my_array,(n,m)))

my_array=numpy.power(a,b)
# 형 변환
my_array=numpy.array(my_array,dtype='int64')
# reshape 함수를 이용
print(numpy.reshape(my_array,(n,m)))
