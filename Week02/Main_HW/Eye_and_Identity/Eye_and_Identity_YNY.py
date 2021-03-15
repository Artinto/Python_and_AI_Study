import numpy
numpy.set_printoptions(sign=' ') # 부호 위치에 공백

n, m = input().split()
print(numpy.eye(int(n), int(m), k = 0))
