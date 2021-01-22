import numpy as np
    
n,m = map(int, input().split())
list1 = []
arr = []
k=0
line=1
for i in range(n):
    for j in range(m):
        if(i == j):
            arr.append(1)
        else:
            arr.append(0)
            
    list1.append(arr)
    arr= []

list1 = np.array(list1)
print(list1)
