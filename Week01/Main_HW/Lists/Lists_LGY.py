# 7가지의 기능 동작하도록 구현하기
def METHOD(method): # ex) method = ['insert', '0', '5']
    
    if method[0] == 'insert':
        List.insert(int(method[1]),int(method[2])) # ex) insert(0, 5)
    elif method[0] == 'remove': 
        List.remove(int(method[1]))
    elif method[0] == 'append':
        List.append(int(method[1]))
    elif method[0] == 'sort':
        List.sort()
    elif method[0] == 'pop':
        List.pop()
    elif method[0] == 'reverse':
        List.reverse()
    elif method[0] == 'print':
        print(List)

    return 0

if __name__ == '__main__':
    N = int(input()) 
    List = []
    for i in range(N):
        method = input().split()  # ex) ['insert', '0', '5']
        METHOD(method)        

           



    

