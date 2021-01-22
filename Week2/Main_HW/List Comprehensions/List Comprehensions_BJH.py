import numpy as np
    
x = int(input())
y = int(input())
z = int(input())
n = int(input())
list1 = []
arr = []
for i in range(x+1):
    for j in range(y+1):
        for k in range(z+1):
            if(i+j+k !=n):
                arr.append(i)
                arr.append(j)
                arr.append(k)
                list1.append(arr)
                arr=[]

list1 = np.array(list1)
print(list1)
