if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    max1=-101
    max2=-101
    for i in arr:
        if max1 < i:
            max1 = i
    for i in arr:
        if i < max1 and max2 < i:
            max2 = i
    print(max2)
