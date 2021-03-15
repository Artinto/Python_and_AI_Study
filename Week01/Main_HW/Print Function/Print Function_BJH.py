if __name__ == '__main__':
    
    n = int(input())
    digit=1
    total=0
    while True:
        total = digit * n + total
        n-=1
        digit =  digit *10
        
        if(n==0):
            break
    print(total) 
       
    
    




