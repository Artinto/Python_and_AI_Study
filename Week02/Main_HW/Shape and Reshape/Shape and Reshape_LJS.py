import numpy as np

Nlist = list(map(int, input().split()))

# Nlist type : list [1,2,3,4,5,6,7,8,9]

Nlist = np.array(Nlist)  

# Nlist type : np.array ( [1,2,3,4,5,6,7,8,9] )

# shape 함수를 사용하기 위해 Type 변환 필요 : list -> Numpy array 

Nlist.shape = (3,3)

print(Nlist)
