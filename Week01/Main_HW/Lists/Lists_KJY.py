if __name__ == '__main__':
    N = int(input())
    a=[]
    for i in range(N):
        s,*arg=input().split()
        argv=list(map(int,arg))
        
        if(s=="insert"):
            a.insert(argv[0],argv[1])
        if(s=="print"):
            print(a)
        if(s=="remove"):
            a.remove(argv[0])
        if(s=="append"):
            a.append(argv[0])
        if(s=="sort"):
            a.sort()
        if(s=="pop"):
            a.pop()
        if(s=="reverse"):
            a.reverse()   
