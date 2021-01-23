if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())

    iarr = [i for i in range(0,x+1)]
    jarr = [j for j in range(0,y+1)]
    karr = [k for k in range(0,z+1)]

    allList = [[i,j,k] for i in iarr for j in jarr for k in karr if i+j+k != n]

    print(allList)
