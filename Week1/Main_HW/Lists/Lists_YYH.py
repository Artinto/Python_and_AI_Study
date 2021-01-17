if __name__ == '__main__':
    N = int(raw_input())
    

arr = []
for t in range(N):
    args = raw_input().strip().split(" ")
    if args[0] == "append":
        arr.append(int(args[1]))
    elif args[0] == "insert":
        arr.insert(int(args[1]), int(args[2]))
    elif args[0] == "remove":
        arr.remove(int(args[1]))
    elif args[0] == "pop":
        arr.pop()
    elif args[0] == "sort":
        arr.sort()
    elif args[0] == "reverse":
        arr.reverse()
    elif args[0] == "print":
        print(arr)
