if __name__ == '__main__':
    N = int(input())
    
    arr = []
    
    for i in range(N):
        command, *Num = input().split()
        Number = list(map(int, Num))
        if(command == 'insert'):
            arr.insert(Number[0],Number[1])
        if(command == 'print'):
            print(arr)
        if(command == 'remove'):
            arr.remove(Number[0])
        if(command == 'append'):
            arr.append(Number[0])
        if(command == 'sort'):
            arr.sort()
        if(command == 'pop'):
            arr.pop()
        if(command == 'reverse'):
            arr.reverse()
