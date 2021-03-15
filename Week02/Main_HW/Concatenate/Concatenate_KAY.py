import numpy as np
n, m, p=input().split()
list1=[]
list2=[]
for i in range(int(n)):
    arr=input().split()
    list1.append(list(map(int, arr)))
    
for i in range(int(m)):
    arr=input().split()
    list2.append(list(map(int, arr)))

print(np.concatenate((list1, list2)))
