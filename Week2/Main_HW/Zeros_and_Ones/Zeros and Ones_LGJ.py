import numpy as np

a,b,c = map(int,input().split())

print(np.zeros((a,b,c), dtype = np.int))

print(np.ones((a,b,c), dtype = np.int))
