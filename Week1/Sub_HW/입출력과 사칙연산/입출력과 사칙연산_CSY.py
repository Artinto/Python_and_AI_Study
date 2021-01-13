#1번
print("Hello World!")

#2번
print("강한친구 대한육군\n강한친구 대한육군")

#3번
print("\\    /\\\n )  ( \')\n(  /  )\n \\(__)|")

#4번
print("|\\_/|\n|q p|   /}\n( 0 )\"\"\"\\\n|\"^\"`    |\n||_/=\\\\__|")

#5번
a, b = map(int,input().split())
print(a+b)

#6번
a, b = map(int,input().split())
print(a-b)

#7번
a, b = map(int,input().split())
print(a*b)

#8번
a, b = map(int,input().split())
print(a/b)

#9번
a, b = map(int,input().split())
print(a+b)
print(a-b)
print(a*b)
print(a//b)
print(a%b)

#10번
a, b, c = map(int,input().split())
print((a+b)%c)
print(((a%c) + (b%c))%c)
print((a*b)%c)
print(((a%c) * (b%c))%c)

#11번
a= int(input())
b= int(input())
print(a*(b%10))
print(a*((b//10)%10))
print(a*(b//100))
print(a*b)
