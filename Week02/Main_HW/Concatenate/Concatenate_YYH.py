import numpy


N,M,P=map(int,input().split())

arr = [list(map(int, input().split())) for _ in range(N+M)]


arr=numpy.concatenate(arr)  

arr.shape=(N+M,P)

print(arr)
