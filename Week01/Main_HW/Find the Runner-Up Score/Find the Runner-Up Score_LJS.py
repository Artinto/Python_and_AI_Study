if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    arr = list(arr)
    
    # Old MAX Value
    MAX = max(arr)
    
    # Remove MAX Value 
    while(MAX in arr):
        arr.remove(MAX)
            
    # New MAX Value
    MAX2 = max(arr)
    
    print(MAX2)
