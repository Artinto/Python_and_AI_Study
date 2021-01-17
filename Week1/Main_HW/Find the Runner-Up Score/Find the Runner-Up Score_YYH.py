if __name__ == '__main__':
    n = int(raw_input())
    arr = map(int, raw_input().split())
    second = largest = -10000
    for i in arr:
        if i > largest:
            second = largest
            largest = i
        elif second < i < largest:
            second = i
            

    print(second)
