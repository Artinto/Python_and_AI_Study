import numpy as np

N,M,P = map(int,input().split())
a = []
b = []
for i in range(N):
    a.append(list(map(int,input().split())))
for i in range(M):
    b.append(list(map(int,input().split())))
array_1 = np.array(a)
array_2 = np.array(b)
print (np.concatenate((array_1, array_2), axis = 0))
