if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    ordered_list = sorted(list(set(arr)))
    print(ordered_list)
    print(ordered_list[-2])

