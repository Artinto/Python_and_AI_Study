import numpy

N = int(input())

array_ = []

for i in range(2): # array A, B >> 2번 반복
    arr = []
    for _ in range(N): # N줄 입력받아서 arr에 저장
        a = numpy.array(input().split(), int)
        arr.append(a)
    array_.append(arr) # 각 A(n개), B(n개)를 array_라는 리스트에 저장

A = numpy.array(array_[0]) # A
B = numpy.array(array_[1]) # B

print (numpy.dot(A, B))
