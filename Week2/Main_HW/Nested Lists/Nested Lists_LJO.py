if __name__ == '__main__':
    list1=[]
    list3=[]
    for _ in range(int(input())):
        name = input()
        score = float(input())
        list2=[name,score]
        list1.append(list2)
        list3.append(list2[-1])
    list3.sort()
    a=min(list3)
    while a in list3:
        list3.remove(a)
    b=list3[0]
    list3.clear()
    for i in list1:
        if i[-1]==b:
            list3.append(i[0])
    list3.sort()
    for i in list3:
        print(i)
