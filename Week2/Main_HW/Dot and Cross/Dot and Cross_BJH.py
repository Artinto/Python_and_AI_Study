import numpy as np
    
n = int(input())
list1 = []
list2 = []
for i in range(n):
    arr=list(map(int,input().split()))
    list1.append(arr)
for i in range(n):
    arr=list(map(int,input().split()))
    list2.append(arr)
    
list1=np.array(list1)
list2=np.array(list2)
print(np.dot(list1,list2))
