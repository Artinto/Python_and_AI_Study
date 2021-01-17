if __name__ == '__main__':
    N = int(input())
    lst = []
    for _ in range(N):
        command, *params = input().split()#이상태로 인수 지정
        if command == 'insert':
            i, e = map(int,params)
            lst.insert(i, e)
            continue
        if command == 'print':
            print( lst )
            continue
        if command == 'remove':
            e = int(params[0])
            lst.remove(e)
            continue
        if command == 'append':
            e = int(params[0])
            lst.append(e)
            continue
        if command == 'sort':
            lst.sort()
            continue
        if command == 'pop':
            lst.pop()
            continue
        if command == 'reverse':
            lst.reverse()
            continue
