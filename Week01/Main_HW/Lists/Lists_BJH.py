if __name__ == '__main__':
    lists= []
    n = int(input())
    while n>0:
        Command = input().split()
        if Command[0] =="""insert""" :
            lists.insert(int(Command[1]), int(Command[2]))
            
        elif Command[0] == """append""" :
            lists.append(int(Command[1]))
            
        elif Command[0] == """remove""" :
            lists.remove(int(Command[1]))
            
        elif Command[0] == """sort""" :
            lists.sort()
            
        elif Command[0] == """pop""" :
            lists.pop()
            
        elif Command[0] == """reverse""" :
            lists.sort(reverse=True)
            
            
        elif Command[0] == """print""" :
            print(lists)

        Command = None
        n-=1
        
    
       
    
    




