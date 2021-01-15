if __name__ == '__main__':#리스트로 풀어봄
    n = int(input())
    arr = list(map(int, input().split()))
check=max(arr)
arr.sort()#오름차순정렬
while arr[len(arr)-1]==check:
    arr.pop()
print(arr[len(arr)-1])

