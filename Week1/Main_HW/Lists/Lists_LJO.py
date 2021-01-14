if __name__ == '__main__':
    n=int(input())
    l=[]
    for i in range(n):
        cmd, *input_data=input().split()
        v=list(map(int,input_data))
        if len(v)>0:
            if cmd=="insert":
                eval("l.{}({}, {})".format(cmd, v[0],v[1]))
            else:
                eval("l.{}({})".format(cmd, v[0]))
        elif cmd=="print":
                print(l)
        else:
            eval("l.{}()".format(cmd))    
