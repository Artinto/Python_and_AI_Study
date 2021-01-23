if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
a=[]
b=[]
for i in range (x+1):
    for j in range (y+1):
        for z in range(z+1):
            if i+j+z !=n:
               b=[i,j,z]
               a.append(b)
               b=[]
print(a,end="")
