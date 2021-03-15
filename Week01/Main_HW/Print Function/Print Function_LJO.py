if __name__ == '__main__':
    n = int(input())
    s=""
    if(0<n<=150):
        for i in range(1,n+1):
            s+=str(i)
    print(s)
