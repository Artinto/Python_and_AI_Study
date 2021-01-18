if __name__ == '__main__':
    N = int(input())
    s = []
    for i in range(N):
        key, *arg = input().split()
        arg = map(int, arg)
        
        if key == 'insert': s.insert(*arg)
        elif key == 'print': print(s)
        elif key == 'remove': s.remove(*arg)
        elif key == 'append': s.append(*arg)
        elif key == 'sort': s.sort()
        elif key == 'pop': s.pop()
        elif key == 'reverse': s.reverse()
        else: pass
        
