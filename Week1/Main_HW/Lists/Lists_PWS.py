if __name__ == '__main__':
    N = int(input())
    arr = []
    for _ in range(N):
        order, *nums = input().split()
        nums = list(map(int, nums))
        if order == 'insert':
            arr.insert(nums[0], nums[1])
        elif order == 'print':
            print(arr)
        elif order == 'remove':
            arr.remove(nums[0])
        elif order == 'sort':
            arr.sort()
        elif order == 'pop':
            arr.pop()
        elif order == 'reverse':
            arr.reverse()
        elif order == 'append':
            arr.append(nums[0])
