import numpy

n,m,p=input().split()
a=[]
for i in range(int(n)):
    a.append(list(map(int,input().split())))
b=[]
for i in range(int(m)):
    b.append(list(map(int,input().split())))
print(numpy.concatenate((a,b)))
