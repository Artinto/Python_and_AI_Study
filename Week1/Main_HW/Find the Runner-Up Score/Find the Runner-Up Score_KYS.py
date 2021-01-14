if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    arr2 = sorted(list(set(arr)))
    print(arr2[-2])