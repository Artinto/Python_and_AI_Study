if __name__ == '__main__':
    N = int(input())

    Arr = []
    
    for i in range(N):
        Command = list(input().split())
        Command[0] = str(Command[0])
        
        if(Command[0] == 'insert'):        
            Command[1] = int(Command[1])
            Command[2] = int(Command[2])
            Arr.insert(Command[1],Command[2])
        elif(Command[0] == 'print'):
            print(list(Arr))
        elif(Command[0] == 'remove'):
            Command[1] = int(Command[1])
            if Command[1] not in Arr:
                continue
            else:
                Arr.remove(Command[1])
        elif(Command[0] == 'append'):
            Command[1] = int(Command[1])
            Arr.append(Command[1])
        elif(Command[0] == 'sort'):
            Arr.sort()
        elif(Command[0] == 'pop'):
            if not Arr:
                continue
            else:
                Arr.pop()
        elif(Command[0] == 'reverse'):
            Arr.reverse()
        else:
            None
        
        
    
        
