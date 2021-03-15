import numpy as np

a = list(map(int,input().split()))
s = np.array(a)
print(np.reshape(s,(3,3)))
