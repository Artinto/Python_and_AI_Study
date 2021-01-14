if __name__ == '__main__':
    
    n=input()
    arr = list(map(int, input().split()))
    temp =0  
    while 1:
        
        if len(arr) == int(n): #int(n) 안하면 n의 형태를 모르기에 else로 넘어간다.
            
            for i in arr:
                if temp <i:
                    temp = i
            arr.remove(temp)
            temp =0
            for i in arr:
                if temp <i:
                    temp = i
            print(temp)
            break;

        else:
           
            print("""처음 입력하신 수 만큼의 갯수를 입력하세요.""")
            n=input()
            arr = list(map(int, input().split()))
            temp =0 



