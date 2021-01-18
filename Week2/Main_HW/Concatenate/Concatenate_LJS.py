import numpy as np

L = list(map(int,input().split()))

N,M,P = L
Mat1 = []
Mat2 = []

for _ in range(N): 
    Mat1.append(list(map(int,input().split()))) 

for _ in range(M): 
    Mat2.append(list(map(int,input().split()))) 
    
Mat1 = np.array(Mat1) # type : list -> np.array
Mat2 = np.array(Mat2) # type : list -> np.array

Mat = np.concatenate( (Mat1, Mat2), axis = 0)

print(Mat)
