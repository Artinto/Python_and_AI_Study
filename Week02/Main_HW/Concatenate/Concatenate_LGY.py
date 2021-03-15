import numpy

N, M, P = map(int, input().split())


arr = []
arr2 = []


for _ in range(N): # NXP
    p = numpy.array(input().split(),int)
    arr.append(p)

array_1 = numpy.array(arr)

for _ in range(M): # MXP
    p = numpy.array(input().split(),int)
    arr2.append(p)

array_2 = numpy.array(arr2)

ans = numpy.concatenate((array_1, array_2), axis = 0)   
# axis = 0 : 옆으로 붙이기
# axis = 1 : 밑으로 붙이기
# (3차원일 경우) axis = 2 : 뒤로 붙이기

print(ans)


