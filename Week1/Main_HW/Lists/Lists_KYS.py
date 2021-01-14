if __name__ == '__main__':
    N = int(input())
    arr=[]
    for _ in range(N):
        arr2 = input().split(" ")
        if arr2[0] == "insert" : arr.insert(int(arr2[1]), int(arr2[2]))
        elif arr2[0] == "print" : print(arr)
        elif arr2[0] == "remove" : arr.remove(int(arr2[1]))
        elif arr2[0] == "append" : arr.append(int(arr2[1]))
        elif arr2[0] == "sort" : arr.sort()
        elif arr2[0] == "pop" : arr.pop()
        else : arr.reverse()