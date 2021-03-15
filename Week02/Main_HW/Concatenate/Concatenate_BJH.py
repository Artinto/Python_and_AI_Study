import numpy as np
    
n,m,p = map(int,input().split())
list1 = []
list2 = []
for i in range(n):
    arr=input().split()
    list1.append(arr)

for i in range(m):
    arr=input().split()
    list2.append(arr)

print(np.concatenate((list1, list2)))
