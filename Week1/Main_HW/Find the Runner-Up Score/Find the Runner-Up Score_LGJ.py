if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    
    A = list(arr)
    
    max = max(A)
    
    runnerup = -1
    
    for i in range(0,len(A)):
        if A[i] < max:
            if A[i] > runnerup: 
                runnerup = A[i]
        elif A[i] == max:
            continue
        
    print(runnerup)
