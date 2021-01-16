a_list=[]
n=int(input())
for i in range(0, n):
    a=input()
    b=a.split()
    if b[0]=="insert":
        a_list.insert(int(b[1]), int(b[2]))
    elif b[0]=="print":
        print(a_list)
    elif b[0]=="remove":
        a_list.remove(int(b[1]))
    elif b[0]=="append":
        a_list.append(int(b[1]))
    elif b[0]=="sort":
        a_list.sort()
    elif b[0]=="pop":
        a_list.pop()
    elif b[0]=="reverse":
        a_list.reverse()
        
