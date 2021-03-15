n=int(input())
A=input()
A=A.split()
F_max=-100000
S_max=-100000
for i in range(0, n):
    if F_max<int(A[i]):
        S_max=F_max
        F_max=int(A[i])
    elif S_max<int(A[i])<F_max:
        S_max=int(A[i])
                                
print(S_max)       
