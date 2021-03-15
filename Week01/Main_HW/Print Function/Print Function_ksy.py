'''if __name__ == '__main__':
    n = int(input())
def func_print(n):
    a=0
    for i in range(1,n+1):    
       a=a+i*(10**(n-i))
    print(a)
func_print(n)'''




if __name__ == '__main__':
    n = int(input())
    for i in range(1,n+1):
        print(i,end='')
