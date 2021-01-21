a_list=[]
b_list=[]
if __name__ == '__main__':
    Result=False
    n=int(input())
    for _ in range(n):
        name=input()
        score=float(input())
        a_list.append([name, score])
    for i in range(n):
        b_list.append(a_list[i][1])
    b_list.sort()
    low=min(b_list)
    while low in b_list:
        b_list.remove(low)
    if n!=3:
        for i in reversed(range(n)):
            if min(b_list)==a_list[i][1]:
                print(a_list[i][0])
    else:
        for i in range(n):
            if min(b_list)==a_list[i][1]:
                print(a_list[i][0])
