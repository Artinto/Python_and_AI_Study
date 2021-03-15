import numpy as np
np.set_printoptions(legacy='1.13')#출력 형식을 맞추어줌
a,b=map(int, input().split())
print(np.eye(a, b))
