if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())

    ans = []

    for i in range(x+1): # 0~x까지
        for j in range(y+1): # 0~y까지
            for k in range(z+1): # 0~z까지
                if i+j+k != n :
                    ans.append([i,j,k])
    print(ans)

