if __name__ == '__main__':
    s=[]
    for _ in range(int(input())):
        name = input()
        score = float(input())
        s.append([name,score])
    name=[]
    score=[]
    
    # 정렬을 위해 name과 score 위치 변경
    for i in range(len(s)):
        tmp=s[i][0]
        s[i][0]=s[i][1]
        s[i][1]=tmp
    
    # 오름차순 정렬 하면 첫번째 요소의 점수가 최솟값
    s.sort()
    mini=s[0][0]
    
    while True:
        if s[0][0]==mini:
            s.remove(s[0])
        else:
            break
    
    # 새로운 리스트의 최솟값
    mini=s[0][0]
    lst=[]
    for i in range(len(s)):
        if s[i][0]==mini:
            lst.append(s[i][1])
    lst.sort()
    for i in range(len(lst)):
        print(lst[i])
