import numpy as np

# Floor division -> 정수부분만 남기고 소수는 버린다.

N,M = list(map(int,input().split()))

MatA = []
MatB = []

for _ in range(N):
    M1 = list(map(int,input().split()))
    MatA.append(M1)

for _ in range(N):
    M2 = list(map(int,input().split()))
    MatB.append(M2)
    
MatA = np.array(MatA) # Type : list -> np.array
MatB = np.array(MatB) # Type : list -> np.array

print(np.add(MatA,MatB))
print(np.subtract(MatA,MatB))
print(np.multiply(MatA,MatB))
print(MatA//MatB)
print(np.mod(MatA,MatB))
print(np.power(MatA,MatB))
