import numpy as np

n, m = input().split()
n = int(n)

a = list(map(int, input().split()))
a = np.array([a])
for _ in range(n-1):
    new = list(map(int, input().split()))
    new = np.array([new])
    a = np.concatenate((a, new), axis=0)

b = list(map(int, input().split()))
b = np.array([b])

for _ in range(n-1):
    new = list(map(int, input().split()))
    new = np.array([new])
    b = np.concatenate((b, new), axis=0)

print(np.add(a, b))
print(np.subtract(a, b))
print(np.multiply(a, b))
print(np.floor_divide(a, b))
print(np.mod(a, b))
print(np.power(a, b))