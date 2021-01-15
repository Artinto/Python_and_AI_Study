def funtion():
    n=int(input())
    num=0
    for i in range(1, n+1):
        if i/10<1:
            num=num+i*10**(n-i)
        elif i/10>=1 and i/100<1:
           num=num*10+i*10**(n-i)    
        else:
            num=num*100+i*10**(n-i)
       
    return num   
print(funtion())   
