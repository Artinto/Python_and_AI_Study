import numpy#리스트를 입력 받아서 numpy객체로 만듬

s=input()
s=s.split()
s=list(map(int, s))
arr1=numpy.array(s)
arr1=numpy.reshape(arr1, (3, 3))
print(arr1)
