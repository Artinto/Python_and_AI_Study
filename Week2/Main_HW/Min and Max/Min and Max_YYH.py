import numpy as np


arr=[]
N,M=map(int, input().split())

for _ in range(N):
    arr.append(list(map(int,input().split())))
    
    

Min= np.min(arr, axis = 1)  

print(np.max(Min))
