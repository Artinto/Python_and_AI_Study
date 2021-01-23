#1
A,B=map(int,input().split())
if A>B:
    print(">")
elif A==B:
    print("==")
else:
    print("<")
    
 #2
a=int(input())
if a>=90 :
    print("A")
elif a>=80 :
    print("B")
elif a>=70 :
    print("C")
elif a>=60:
    print("D")
else:
    print("F")
    
 #3
year=int(input())
if year%4==0and year%100 or year%400==0:
    print("1")
else:
    print("0")
    
 #4
x=int(input())
y=int(input())
if x>0 :
    if y>0:
        print("1")
    else :
        print("4")
else :
    if y>0:
        print("2")
    else :
        print("3")  
       
 #5
h,m=map(int,input().split())
if h>=1:
    if m>=45:
        m-=45
    else:
        h-=1
        m=(m+60)-45
else:
    if m>=45:
        m-=45
    else:
        h=23
        m=(m+60)-45
print(h,m)
