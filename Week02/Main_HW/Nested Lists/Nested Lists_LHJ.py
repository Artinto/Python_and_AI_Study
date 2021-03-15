if __name__ == '__main__':
    arr = {}
    for _ in range(int(input())):
        name = input()
        score = float(input())
        if score in arr.keys():
            arr[score].append(name)
        else:
            arr[score] = [name]
    
    score = sorted(arr)[1]
    
    print(*sorted(arr[score]), sep='\n')
