import numpy as np
 
N=int(input())

A=[]
B=[]

for _ in range(N):
      
    A.append(list(map(int,input().split())))

for _ in range(N):
      
    B.append(list(map(int,input().split())))


a = np.array(A) 
b = np.array(B) 

 
print(np.dot(a, b))
