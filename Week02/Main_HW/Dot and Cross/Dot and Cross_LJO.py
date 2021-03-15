import numpy as np

n=int(input())
A=[]
B=[]
for _ in range(n):
    l1=list(map(int,input().split()))
    A.append(l1)
for _ in range(n):
    l2=list(map(int,input().split()))
    B.append(l2)
A=np.array(A)
B=np.array(B)
print(np.dot(A,B))
