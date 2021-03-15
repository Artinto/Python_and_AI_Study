import numpy as np
n,m = map(int, input().split())
list1 = []
for _ in  range (n):
    arr = list(map(int, input().split()))
    list1.append(arr)
list1 = np.array(list1)
list1 = np.min(list1,axis=1)
print(np.max(list1))
