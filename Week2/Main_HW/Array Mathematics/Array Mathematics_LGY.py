import numpy

n, m = map(int, input().split())
A = []
B = []

for _ in range(n): # N lines 받아들이기
    arr1 = numpy.array(input().split(),int) # M
    A.append(arr1)
# NXM의 numpy array로 저장됨.    
    
for _ in range(n):
    arr2 = numpy.array(input().split(),int)
    B.append(arr2)


ans1 = numpy.add(A,B)
ans2 = numpy.subtract(A,B)
ans3 = numpy.multiply(A,B)
ans4 = numpy.floor_divide(A,B) # floor : 버림
ans5 = numpy.mod(A,B)
ans6 = numpy.power(A,B)

print(ans1)
print(ans2)
print(ans3)
print(ans4)
print(ans5)
print(ans6)


