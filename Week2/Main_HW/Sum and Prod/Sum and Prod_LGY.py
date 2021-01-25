import numpy as np

n,m = map(int, input().split())
arr = []

for _ in range(n):
    line = np.array(input().split(),int)
    arr.append(line)
    
s = np.sum(arr,axis = 0)
p = np.prod(s)
print(p)



