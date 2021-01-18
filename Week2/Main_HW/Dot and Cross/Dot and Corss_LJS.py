import numpy as np

N = int(input())

MatA = []
MatB = []

for _ in range(N):
    NA_cols_ = list(map(int,input().split()))
    MatA.append(NA_cols_)

for _ in range(N):
    NB_cols_ = list(map(int,input().split()))
    MatB.append(NB_cols_)
    

MatA = np.array(MatA) # Type : list -> np.array
MatB = np.array(MatB) # Type : list -> np.array

Dot = np.dot(MatA, MatB)
print(Dot)


