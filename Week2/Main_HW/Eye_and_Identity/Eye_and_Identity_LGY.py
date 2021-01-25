import numpy as np

np.set_printoptions(sign=' ') # print시 항상 space 출력하기
N , M = map(int, input().split())

print(np.eye(N, M))

