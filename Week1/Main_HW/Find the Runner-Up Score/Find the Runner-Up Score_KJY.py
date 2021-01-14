if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    array=list(arr)
    
    max=array[0]
    next=array[0]
            
    for i in array:
       if i>max:
            next=max
            max=i
       elif max==next:
            if max>i:
                next=i
       elif i>next and i<max:
            next=i
    print(next)
