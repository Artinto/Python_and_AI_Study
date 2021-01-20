if __name__ == '__main__':
    s=[]
    for _ in range(int(input())):
        name = input()
        score = float(input())
        s.append([name,score])
    name=[]
    score=[]
    
    for i in range(len(s)):
        tmp=s[i][0]
        s[i][0]=s[i][1]
        s[i][1]=tmp
    s.sort()
    mini=s[0][0]
    while True:
        if s[0][0]==mini:
            s.remove(s[0])
        else:
            break
    mini=s[0][0]
    lst=[]
    for i in range(len(s)):
        if s[i][0]==mini:
            lst.append(s[i][1])
    lst.sort()
    for i in range(len(lst)):
        print(lst[i])
