#1
if __name__=="__main__":
    print("Hello World!")

#2
if __name__=="__main__":
    print("강한친구 대한육군")
    print("강한친구 대한육군")

#3
if __name__=="__main__":
    print("\\    /\\")
    print(" )  ( ')")
    print("(  /  )")
    print(" \\(__)|")

#4
if __name__=="__main__":
    print("|\\_/|")
    print("|q p|   /}")
    print("( 0 )\"\"\"\\")
    print("|\"^\"`    |")
    print("||_/=\\\\__|")      


#5
if __name__=="__main__":
    A,B=list(map(int,input().split()))
    print(A+B)

#6
if __name__=="__main__":
    A,B=list(map(int,input().split()))
    print(A-B)

#7
if __name__=="__main__":
    A,B=list(map(int,input().split()))
    print(A*B)
    
#8
if __name__=="__main__":
    A,B=list(map(int,input().split()))
    print(A/B)

#9
if __name__=="__main__":
    A,B=list(map(int,input().split()))
    print(A+B)
    print(A-B)
    print(A*B)
    print(A//B)
    print(A%B)

#10
if __name__=="__main__":
    A,B,C=list(map(int,input().split()))
    print((A+B)%C)
    print(((A%C)+(B%C))%C)
    print((A*B)%C)
    print(((A%C)*(B%C))%C)

#11
if __name__=="__main__":
    a=input()
    b=input()
    a=int(a)
    b=int(b)
    n1=a*(b%10)
    n2=a*((b%100)//10)
    n3=a*(b//100)
    print(n1)
    print(n2)
    print(n3)
    print(a*b)
