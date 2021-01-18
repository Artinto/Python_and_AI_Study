if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())	# <map object at 0x7f6f9c3762d0>
    arr = set(arr)			# {2, 3, 5, 6}
    arr = sorted(arr)			# [2, 3, 5, 6]
    print(arr[-2])			# 5
    
    