n=int(input())
marks={}
for i in range(0, n):
    A=input()
    a, b=A.split(' ', 1)
    marks[a]=b
name=input()   
for i in range(0, n):
    if marks.get(name)!=None:
        a, b, c=marks[name].split()
        ave=(float(a)+float(b)+float(c))/3    
print(format(ave, '.2f'))        
